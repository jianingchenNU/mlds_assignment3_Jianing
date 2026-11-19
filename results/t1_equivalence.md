# T1 equivalence

(a) FedAvg C=1, E=1, full batch vs centralized GD, K=100, 10 rounds
(b) FedAvg K=1 (client 0, n=275), E=2, batch 32 vs plain local SGD

| round | (a) max abs dw | (b) max abs dw |
|---|---|---|
| 1 | 7.451e-09 | 0.000e+00 |
| 2 | 7.451e-09 | 0.000e+00 |
| 3 | 1.490e-08 | 0.000e+00 |
| 4 | 1.490e-08 | 0.000e+00 |
| 5 | 2.235e-08 | 0.000e+00 |
| 6 | 2.235e-08 |  |
| 7 | 2.608e-08 |  |
| 8 | 2.980e-08 |  |
| 9 | 3.353e-08 |  |
| 10 | 3.725e-08 |  |
