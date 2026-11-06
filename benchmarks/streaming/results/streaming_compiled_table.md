### Per-tick cost (microseconds), 1 channel + time augmentation + lead-lag

| depth | window | streaming | streaming (auto refresh) | compiled | compiled (auto refresh) | batch `numpy` | batch `iisignature` | batch `roughpy` | x vs numpy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 10 | 104.7 | 130.7 | 3.2 | 3.1 | 380.6 | 17.4 | 192.4 | 4x |
| 2 | 30 | 110.4 | 142.8 | 2.6 | 2.7 | 1142.2 | 20.4 | 376.0 | 10x |
| 2 | 100 | 105.5 | 146.6 | 2.7 | 2.8 | 3922.8 | 32.2 | 1047.4 | 37x |
| 2 | 300 | 109.2 | 146.4 | 2.8 | 2.9 | 12447.0 | 63.4 | 2915.8 | 114x |
| 2 | 600 | 110.6 | 142.4 | 2.8 | 3.0 | 25229.6 | 111.8 | 6225.6 | 228x |
| 2 | 1200 | 113.8 | 161.5 | 3.2 | 3.3 | 53515.8 | 234.5 | 12438.5 | 470x |
| 3 | 10 | 162.1 | 218.8 | 3.6 | 3.9 | 633.8 | 22.2 | 464.4 | 4x |
| 3 | 30 | 179.2 | 237.8 | 3.5 | 3.8 | 1975.7 | 33.6 | 884.8 | 11x |
| 3 | 100 | 183.2 | 241.7 | 3.5 | 3.8 | 7223.2 | 73.3 | 2381.3 | 39x |
| 3 | 300 | 187.1 | 261.6 | 3.7 | 4.0 | 21449.8 | 174.5 | 6358.7 | 115x |
| 3 | 600 | 188.5 | 251.2 | 3.9 | 4.1 | 42923.6 | 342.6 | 13428.3 | 228x |
| 3 | 1200 | 191.2 | 250.9 | 3.9 | 4.3 | 91899.4 | 668.8 | 26688.5 | 481x |

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

### Compiled engine drift (relative, infinity norm)

| depth | refresh every | relative error | worst level | re-anchors |
| --- | ---: | ---: | ---: | ---: |
| 2 | never | 5.55e-15 | 5.55e-15 (level 1) | 0 |
| 2 | 250 | 5.55e-15 | 5.55e-15 (level 1) | 78 |
| 3 | never | 5.25e-14 | 3.13e-13 (level 3) | 0 |
| 3 | 250 | 5.55e-15 | 5.55e-15 (level 1) | 78 |
| 4 | never | 3.90e-12 | 9.29e-11 (level 4) | 0 |
| 4 | 250 | 5.55e-15 | 5.55e-15 (level 1) | 78 |
| 5 | never | 2.27e-10 | 2.69e-08 (level 5) | 0 |
| 5 | 250 | 5.55e-15 | 6.82e-15 (level 5) | 78 |

### Nested 600/300/150 windows, per ORVP segment (milliseconds)

| route | ms/segment |
| --- | ---: |
| naive: three independent signatures (numpy) | 74.02 |
| naive: three independent signatures (iisignature) | 0.65 |
| Chen: disjoint chunks combined | 45.63 |
