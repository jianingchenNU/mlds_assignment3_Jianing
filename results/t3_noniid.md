# T3 natural vs IID split

C = 0.1, 100 rounds, seed 0

| E | split | final val acc | mean drift |
|---|---|---|---|
| 1 | natural | 0.4803 | 0.2375 |
| 1 | iid | 0.4800 | 0.1783 |
| 5 | natural | 0.6569 | 0.9037 |
| 5 | iid | 0.6763 | 0.7203 |
| 10 | natural | 0.6967 | 1.4576 |
| 10 | iid | 0.7368 | 1.2660 |

(1) IID > natural val acc for every E: False
(2) drift increasing in E for both splits: True
(3) drift natural > IID for every E: True
