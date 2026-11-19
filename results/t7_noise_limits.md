# T7 noise limits

source: test_results.json

sigma_x = 0.1589; b* = 0.1124; majority-class test rate 0.0610; halfway accuracy 0.4100 reached at b = 0.8960

| b | test acc | change vs previous b |
|---|---|---|
| 0 | 0.7589 |  |
| 0.05 | 0.7462 | -0.0127 |
| 0.1 | 0.7233 | -0.0229 |
| 0.2 | 0.6589 | -0.0643 |
| 0.3 | 0.6117 | -0.0472 |
| 0.5 | 0.5391 | -0.0726 |
| 0.75 | 0.4435 | -0.0956 |
| 1 | 0.3861 | -0.0574 |
| 2 | 0.2447 | -0.1414 |
| 10 | 0.0226 | -0.2220 |

(a) non-increasing within 0.01: True; failing pairs: none
(b) acc(b=10) = 0.0226 <= 0.0610 + 0.03: True
