# T6 Laplace noise

Noise measured on the training split of 100 clients (eps = X' - X).

| b | samples | mean | E abs eps (= b) | Var (= 2b^2) | KS p | share outside [0,1] | same seed same copy | pass |
|---|---|---|---|---|---|---|---|---|
| 0.1 | 19964560 | -2.95e-05 | 0.10000 (+0.00%) | 0.02000 (+0.00%) | 0.722 | 0.476 | True | True |
| 0.5 | 19964560 | -1.48e-04 | 0.50001 (+0.00%) | 0.50002 (+0.00%) | 0.722 | 0.560 | True | True |
| 1.0 | 19964560 | -2.95e-04 | 1.00001 (+0.00%) | 2.00007 (+0.00%) | 0.722 | 0.681 | True | True |

b = 0: laplace_perturb returns X bitwise unchanged: True; 5-round history with perturb_clients(b=0) identical to plain run: True
