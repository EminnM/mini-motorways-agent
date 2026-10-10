### S1 results (3 training seeds per architecture; synthetic Gap-1, immediate completion reward, native-compatible 3,111-action space)

Held-out TEST (400 unseen layouts), checkpoint SELECTED on VALIDATION only (deterministic policy unless stated); mean ± sd across seeds.

| policy | test completion | first-action accuracy | median steps | destructive | timeout | WAIT / ADD / REMOVE | P(correct) at decision 1 | median rank of correct action |
|---|---|---|---|---|---|---|---|---|
| WAIT | 0.0% | 0.0% | None | 0.0% | 100.0% | 100 / 0 / 0 % | - | - |
| RANDOM | 9.3% | 0.5% | 10 | 11.1% | 79.6% | 1 / 91 / 8 % | - | - |
| ORACLE | 100.0% | 100.0% | 1.0 | 0.0% | 0.0% | 0 / 100 / 0 % | - | - |
| flat MLP (selected ckpt, deterministic) | 6.8% ± 0.3 | 0.2% ± 0.0 | [7.0, 11.0, 8.5] | 9.9% ± 4.7 | 83.2% ± 4.4 | 6 / 83 / 11 % | 0.0046 | 99.3 |
| flat MLP (selected ckpt, stochastic) | 9.1% ± 1.2 | 0.8% ± 0.0 | [7.0, 9, 12.5] | 14.4% ± 2.7 | 76.5% ± 1.7 | 1 / 91 / 8 % | 0.0046 | 99.3 |
| flat MLP (final ckpt, deterministic) | 7.7% ± 0.8 | 0.4% ± 0.3 | [6.0, 11.0, 8.5] | 10.2% ± 2.2 | 82.1% ± 3.0 | 0 / 83 / 17 % | 0.0048 | 98.0 |
| spatial CNN (selected ckpt, deterministic) | 89.8% ± 6.1 | 39.6% ± 4.7 | [2.0, 2, 2] | 0.6% ± 1.0 | 9.6% ± 6.3 | 5 / 94 / 1 % | 0.2533 | 2.7 |
| spatial CNN (selected ckpt, stochastic) | 90.5% ± 2.6 | 25.9% ± 1.9 | [3, 3, 3.0] | 0.5% ± 0.9 | 9.0% ± 2.4 | 5 / 94 / 1 % | 0.2533 | 2.7 |
| spatial CNN (final ckpt, deterministic) | 89.2% ± 6.9 | 39.6% ± 4.7 | [2.0, 2, 2] | 1.1% ± 1.9 | 9.7% ± 7.3 | 5 / 94 / 1 % | 0.2530 | 2.3 |

Per-seed TEST completion (selected checkpoint, deterministic): MLP [0.07, 0.07, 0.065], CNN [0.835, 0.902, 0.958].

### Validation completion by training step (200 validation layouts; mean ± sd over seeds)

| decisions | MLP det. | MLP stoch. | CNN det. | CNN stoch. |
|---|---|---|---|---|
| 0 | 7.5% ± 1.8 | 9.0% ± 1.8 | 1.7% ± 2.9 | 8.0% ± 1.8 |
| 10000 | 6.7% ± 1.2 | 8.3% ± 1.0 | 27.0% ± 23.4 | 22.7% ± 15.3 |
| 20000 | 6.3% ± 1.3 | 7.8% ± 1.2 | 67.3% ± 4.9 | 69.8% ± 0.6 |
| 30000 | 7.2% ± 0.8 | 9.3% ± 0.8 | 81.0% ± 3.8 | 82.7% ± 3.6 |
| 40000 | 8.2% ± 1.2 | 10.3% ± 3.0 | 84.7% ± 2.8 | 88.0% ± 1.3 |
| 50000 | 9.0% ± 1.3 | 9.0% ± 0.0 | 89.5% ± 4.8 | 92.2% ± 4.6 |
| 50176 | 9.2% ± 2.1 | 8.7% ± 0.8 | 89.7% ± 5.5 | 91.5% ± 3.5 |

Validation baselines: WAIT 0.0%, RANDOM LEGAL 9.8% (5 seeds), ORACLE 100%.

### Cost
| architecture | parameters | training SPS (per process, 6 processes in parallel) | wall-clock for 50k decisions | native game reference |
|---|---|---|---|---|
| mlp | 1,703,976 | 96 | 8.7 min | native ~4-5 SPS (about 3 h per 50k) |
| cnn | 73,416 | 30 | 27.7 min | native ~4-5 SPS (about 3 h per 50k) |

### Memorisation checks (TEST)

- Most common correct-action index covers 0.8% of test tasks, so repeating one index cannot exceed that.
- mlp: tasks whose correct action index never occurs in TRAIN: n=70; completion on them per seed [0.114, 0.071, 0.029]
- cnn: tasks whose correct action index never occurs in TRAIN: n=70; completion on them per seed [0.814, 0.857, 0.943]
