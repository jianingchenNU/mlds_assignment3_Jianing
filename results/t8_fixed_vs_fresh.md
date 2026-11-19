# T8 fixed vs fresh noise

b = 0.5, C = 0.1, E = 5, 100 rounds, seed 0

| mode | clean test acc (primary) | final train acc | final val acc (noisy) | m (releases per image) | epsilon_total = m/b per pixel |
|---|---|---|---|---|---|
| fixed | 0.5278 | 0.4746 | 0.2256 | 1 | 2 |
| fresh | 0.5639 | 0.3334 | 0.3264 | mean 50.00 over all clients, 50.00 over selected, max 85 | mean 100, max 170 |

clients selected at least once: 100 of 100
