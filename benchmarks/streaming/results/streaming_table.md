### Per-tick cost (microseconds), 1 channel + time augmentation + lead-lag

| depth | window | streaming | streaming (auto refresh) | batch `numpy` | batch `iisignature` | batch `roughpy` | x vs numpy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 10 | 127.0 | 157.4 | 458.8 | 22.4 | 216.3 | 4x |
| 2 | 30 | 129.4 | 168.9 | 1383.6 | 25.6 | 426.7 | 11x |
| 2 | 100 | 127.7 | 171.8 | 4726.2 | 37.9 | 1163.9 | 37x |
| 2 | 300 | 130.3 | 173.4 | 14146.0 | 71.8 | 3280.7 | 109x |
| 2 | 600 | 132.0 | 170.6 | 28542.2 | 125.1 | 6546.3 | 216x |
| 2 | 1200 | 135.0 | 178.8 | 60000.1 | 231.4 | 13291.2 | 444x |
| 3 | 10 | 203.0 | 249.5 | 722.5 | 27.1 | 528.0 | 4x |
| 3 | 30 | 200.6 | 264.0 | 2232.9 | 39.4 | 874.0 | 11x |
| 3 | 100 | 189.9 | 259.9 | 7402.4 | 73.8 | 2374.5 | 39x |
| 3 | 300 | 194.4 | 265.8 | 22483.7 | 172.5 | 6673.0 | 116x |
| 3 | 600 | 197.0 | 260.2 | 45987.6 | 326.4 | 13208.0 | 233x |
| 3 | 1200 | 201.9 | 264.6 | 91302.8 | 628.0 | 25987.6 | 452x |

### Drift vs a from-scratch signature (relative, infinity norm)

| depth | refresh every | relative error | worst level | re-anchors |
| --- | ---: | ---: | ---: | ---: |
| 2 | never | 8.29e-16 | 1.65e-15 (level 2) | 0 |
| 2 | 250 | 6.81e-17 | 1.36e-16 (level 2) | 78 |
| 3 | never | 5.41e-13 | 3.23e-12 (level 3) | 0 |
| 3 | 250 | 6.94e-15 | 4.14e-14 (level 3) | 78 |
| 4 | never | 4.23e-11 | 1.01e-09 (level 4) | 0 |
| 4 | 250 | 6.94e-15 | 1.47e-13 (level 4) | 78 |
| 5 | never | 1.66e-09 | 1.97e-07 (level 5) | 0 |
| 5 | 250 | 6.94e-15 | 3.56e-13 (level 5) | 78 |

### Nested 600/300/150 windows, per ORVP segment (milliseconds)

| route | ms/segment |
| --- | ---: |
| naive: three independent signatures (numpy) | 76.74 |
| naive: three independent signatures (iisignature) | 0.58 |
| Chen: disjoint chunks combined | 46.43 |
