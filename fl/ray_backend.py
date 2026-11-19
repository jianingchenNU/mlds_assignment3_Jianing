"""Ray Core backend: one short-lived actor per selected client per round."""
import time

import ray
import torch
from ray.util.state import list_actors

from fl.client import local_train
from fl.server import fit_x
from fl.utils import ROOT


@ray.remote(num_cpus=1)
class ClientActor:
    def __init__(self, data):
        # 4 actors share 4 cores; more threads each would oversubscribe the CPU
        torch.set_num_threads(1)
        self.data = data

    def train(self, w, E, lr, batch, seed, device, fresh_b, clip):
        return local_train(w, self.data["x_tr"], self.data["y_tr"], E, lr, batch, seed, device, fresh_b, clip)


def start(clients, num_cpus):
    # workers need the repo on sys.path to import fl; the dashboard must stay on
    # because the state API (list_actors) is served by it
    ray.init(num_cpus=num_cpus, runtime_env={"env_vars": {"PYTHONPATH": str(ROOT)}},
             log_to_driver=False)
    # client data goes to the object store once; actors get only the ObjectRef
    return [ray.put({"x_tr": fit_x(c), "y_tr": c["y_tr"]}) for c in clients]


def alive():
    return len(list_actors(filters=[("state", "=", "ALIVE")], limit=1000))


def wait_alive(target, timeout=30.0):
    # ray.kill is asynchronous; the state API catches up within a short time
    t0 = time.perf_counter()
    n = alive()
    while n != target and time.perf_counter() - t0 < timeout:
        time.sleep(0.05)
        n = alive()
    return n


def train_round(w, sel, seeds, refs, args, fresh_b):
    """Create one actor per selected client, train, kill all. Returns (ws, stats, actor row)."""
    before = alive()
    t0 = time.perf_counter()
    actors = [ClientActor.options(num_gpus=args.gpus_per_client).remote(refs[k]) for k in sel]
    w_ref = ray.put(w)
    out = ray.get([a.train.remote(w_ref, args.E, args.lr, args.batch, s, args.device, fresh_b, args.clip)
                   for a, s in zip(actors, seeds)])
    t_train = time.perf_counter() - t0
    during = alive()
    t0 = time.perf_counter()
    for a in actors:
        ray.kill(a)
    after = wait_alive(before)
    t_kill = time.perf_counter() - t0
    row = {"alive_before": before, "created": len(actors), "alive_during": during, "alive_after_kill": after,
           "create_train_s": round(t_train, 4), "kill_wait_s": round(t_kill, 4)}
    return [o[0] for o in out], [o[1] for o in out], row


def stop():
    ray.shutdown()
