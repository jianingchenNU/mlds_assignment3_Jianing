# T2 Ray vs serial

4 clients per round, 5 rounds, seed 0; final max abs dw = 0.000e+00

| round | serial s | ray s | ray create+train s | ray kill wait s | alive before | alive during | alive after kill |
|---|---|---|---|---|---|---|---|
| 1 | 0.759 | 3.725 | 3.6576 | 0.021 | 0 | 4 | 0 |
| 2 | 0.087 | 3.619 | 3.5878 | 0.0186 | 0 | 4 | 0 |
| 3 | 0.099 | 3.584 | 3.5399 | 0.0148 | 0 | 4 | 0 |
| 4 | 0.079 | 3.575 | 3.5572 | 0.0053 | 0 | 4 | 0 |
| 5 | 0.094 | 3.586 | 3.556 | 0.0163 | 0 | 4 | 0 |

mean round time: serial 0.224 s, ray 3.618 s, speed-up serial/ray = 0.06x

## Scaling (3 rounds per E, median round time)

t1 = 0.00601 s, c = 3.519 s, predicted E* = c/(3 t1) = 195.0; measured E* = 301.9

| E | serial s | ray s | serial model s | ray model s |
|---|---|---|---|---|
| 5 | 0.120 | 3.549 | 0.120 | 3.549 |
| 50 | 0.826 | 3.739 | 1.203 | 3.819 |
| 100 | 1.604 | 3.947 | 2.406 | 4.120 |
| 200 | 3.315 | 4.453 | 4.811 | 4.721 |
| 400 | 6.576 | 5.479 | 9.623 | 5.924 |
