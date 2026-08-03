| Arm | Features | seed 0 | seed 1 | seed 2 | Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| naive | 1 | 0.33485 | 0.40147 | 0.39580 | 0.37737 |
| har | 27 | 0.24075 | 0.27261 | 0.27822 | 0.26386 |
| book | 37 | 0.23177 | 0.27035 | 0.26953 | 0.25722 |
| book+har | 63 | 0.23093 | 0.27121 | 0.27218 | 0.25811 |
| sig | 91 | 0.24362 | 0.28669 | 0.28993 | 0.27341 |
| sig+har | 117 | 0.23980 | 0.28040 | 0.28368 | 0.26796 |
| sig+book | 127 | 0.23199 | 0.27928 | 0.27512 | 0.26213 |
| sig+book+har | 153 | 0.23133 | 0.27603 | 0.27497 | 0.26078 |
| multisig | 109 | 0.24192 | 0.28645 | 0.28476 | 0.27104 |
| multisig+har | 135 | 0.23902 | 0.27736 | 0.28419 | 0.26686 |
| multisig+book | 145 | 0.23250 | 0.27317 | 0.27218 | 0.25929 |
| multisig+book+har | 171 | 0.23112 | 0.27478 | 0.27259 | 0.25950 |

| Comparison | seed 0 | seed 1 | seed 2 | Mean | Seeds with p_no_improvement < 0.05 |
| --- | ---: | ---: | ---: | ---: | ---: |
| book+har vs naive | +31.03% | +32.45% | +31.23% | +31.57% | 3/3 |
| sig vs har | -1.19% | -5.17% | -4.21% | -3.52% | 0/3 |
| sig+har vs har | +0.40% | -2.86% | -1.96% | -1.47% | 1/3 |
| sig+book vs book | -0.10% | -3.30% | -2.07% | -1.82% | 0/3 |
| sig+book+har vs book+har | -0.17% | -1.78% | -1.02% | -0.99% | 0/3 |
| multisig vs har | -0.48% | -5.08% | -2.35% | -2.64% | 0/3 |
| multisig+har vs har | +0.72% | -1.75% | -2.14% | -1.06% | 1/3 |
| multisig+book vs book | -0.32% | -1.05% | -0.98% | -0.78% | 0/3 |
| multisig+book+har vs book+har | -0.08% | -1.32% | -0.15% | -0.52% | 0/3 |
| multisig vs sig | +0.70% | +0.08% | +1.78% | +0.85% | 1/3 |
| multisig+book+har vs sig+book+har | +0.09% | +0.45% | +0.87% | +0.47% | 0/3 |

**Verdict:** multichannel signatures do not consistently improve book+har (0/3 seeds with p_no_improvement < 0.05, mean -0.52%) -- stop the ORVP search.
