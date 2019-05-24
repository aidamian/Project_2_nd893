# Continuous Control Report


First iteration with no noise reduction reched a 24.5 average at episode 250 then plateaued at 22-23.
```
First training iter at step 1010
Episode   50  Score/Max/Avg:  5.3/ 7.8/ 2.8  AvStp: 1000  [μcL1/μcL2:  7.4e-04/ 6.9e-04 μaL:  2.7e-01]  
  Loaded steps:      50050
  Train iters:       24521
  Actor update:      12260
Episode  100  Score/Max/Avg: 16.5/16.5/ 5.9  AvStp: 1000  [μcL1/μcL2:  1.5e-03/ 1.5e-03 μaL:  3.6e-01]  
  Loaded steps:     100100
  Train iters:       49550
  Actor update:      24775
Episode  150  Score/Max/Avg: 13.2/36.7/13.0  AvStp: 1000  [μcL1/μcL2:  5.7e-03/ 5.8e-03 μaL:  1.6e-01] 
  Loaded steps:     150150
  Train iters:       74571
  Actor update:      37285
Episode  200  Score/Max/Avg: 22.8/39.3/20.5  AvStp: 1000  [μcL1/μcL2:  1.1e-02/ 1.1e-02 μaL: -9.6e-01]  
  Loaded steps:     200200
  Train iters:       99600
  Actor update:      49800
Episode  250  Score/Max/Avg: 24.1/39.3/24.5  AvStp: 1000  [μcL1/μcL2:  1.8e-02/ 1.6e-02 μaL: -5.2e-01]  
  Loaded steps:     250250
  Train iters:      124621
  Actor update:      62310
Episode  300  Score/Max/Avg: 29.4/39.3/22.9  AvStp: 1000  [μcL1/μcL2:  6.7e-02/ 7.0e-02 μaL:  9.3e-01]  
  Loaded steps:     300300
  Train iters:      149650
  Actor update:      74825
Episode  350  Score/Max/Avg: 21.5/39.5/21.5  AvStp: 1000  [μcL1/μcL2:  3.9e-01/ 3.6e-01 μaL:  2.0e+00]  
  Loaded steps:     350350
  Train iters:      174671
  Actor update:      87335
```

with exploration noise reduction we get similar results:

``` 
First training iter at step 1010
Episode   50  Score/Max/Avg:  6.9/ 6.9/ 2.7  AvStp: 1000  [μcL1/μcL2:  4.4e-04/ 4.4e-04 μaL:  2.6e-01]  
  Loaded steps:      50050
  Train iters:       24521
  Actor update:      12260

New explor noise: 0.0750
Episode  100  Score/Max/Avg: 13.6/19.3/ 5.8  AvStp: 1000  [μcL1/μcL2:  1.5e-03/ 1.4e-03 μaL:  2.8e-01] 
  Loaded steps:     100100
  Train iters:       49550
  Actor update:      24775

New explor noise: 0.0563
Episode  150  Score/Max/Avg: 30.4/34.6/13.1  AvStp: 1000  [μcL1/μcL2:  9.8e-03/ 1.3e-02 μaL:  2.0e-01] 
  Loaded steps:     150150
  Train iters:       74571
  Actor update:      37285

New explor noise: 0.0422
Episode  200  Score/Max/Avg: 25.9/34.8/19.3  AvStp: 1000  [μcL1/μcL2:  9.1e-03/ 8.6e-03 μaL: -6.5e-01] 
  Loaded steps:     200200
  Train iters:       99600
  Actor update:      49800

New explor noise: 0.0316
Episode  250  Score/Max/Avg: 24.5/36.9/25.0  AvStp: 1000  [μcL1/μcL2:  1.7e-02/ 1.5e-02 μaL: -1.3e+00] 
  Loaded steps:     250250
  Train iters:      124621
  Actor update:      62310

New explor noise: 0.0237
Episode  300  Score/Max/Avg: 22.9/36.9/27.0  AvStp: 1000  [μcL1/μcL2:  3.0e-02/ 2.7e-02 μaL: -6.2e-01] 
  Loaded steps:     300300
  Train iters:      149650
  Actor update:      74825

New explor noise: 0.0178
Episode  350  Score/Max/Avg: 20.6/36.9/19.4  AvStp: 1000  [μcL1/μcL2:  1.5e+00/ 1.4e+00 μaL:  1.6e+01] 
  Loaded steps:     350350
  Train iters:      174671
  Actor update:      87335

New explor noise: 0.0133
Episode  400  Score/Max/Avg:  3.5/36.9/11.3  AvStp: 1000  [μcL1/μcL2:  5.8e+00/ 6.1e+00 μaL:  5.0e+01] 
  Loaded steps:     400400
  Train iters:      199700
  Actor update:      99850

```

now with policy noise reduction (this time we introduce a weight debug monitoring procedure):
```
First training iter at step 1010
Episode   50  Score/Max/Avg:  6.4/13.1/ 4.1  AvStp: 1000  [μcL1/μcL2:  7.4e-04/ 6.9e-04 μaL:  2.0e-01]  
  Loaded steps:      50050
  Train iters:       24521
  Actor update:      12260
Model Actor min/max/mean/median:
  layers.0.weight:           -1.9e+00 /  2.3e+00 /  1.9e-02 /  1.6e-02
  layers.0.bias:             -4.8e-01 /  2.3e-01 / -1.4e-01 / -1.4e-01
  layers.2.weight:           -9.7e-01 /  6.6e-01 / -3.5e-02 / -2.7e-02
  layers.2.bias:             -2.8e-01 /  1.7e-01 / -4.0e-02 / -3.0e-02
  final_linear.weight:       -6.4e-01 /  6.0e-01 /  1.1e-02 /  6.4e-03
  final_linear.bias:         -3.9e-02 /  1.1e-01 /  4.7e-02 /  5.9e-02
Model Critic min/max/mean/median:
  final_layers.0.weight:     -1.1e+00 /  6.3e-01 / -4.6e-02 / -3.4e-02
  final_layers.0.bias:       -1.7e-01 /  2.3e-01 /  1.1e-02 /  3.7e-03
  state_layers.0.weight:     -2.0e+00 /  2.6e+00 /  1.3e-02 /  1.1e-02
  state_layers.0.bias:       -4.9e-01 /  1.5e-01 / -1.7e-01 / -1.8e-01
  final_linear.weight:       -7.0e-02 /  1.6e-01 / -1.1e-02 / -2.1e-02
  final_linear.bias:          4.1e-02 /  4.1e-02 /  4.1e-02 /  4.1e-02

New policy noise: 0.1600
Episode  100  Score/Max/Avg: 13.1/20.6/ 8.1  AvStp: 1000  [μcL1/μcL2:  2.9e-03/ 2.6e-03 μaL:  9.1e-02] 
  Loaded steps:     100100
  Train iters:       49550
  Actor update:      24775
Model Actor min/max/mean/median:
  layers.0.weight:           -2.3e+00 /  2.6e+00 /  1.8e-02 /  1.5e-02
  layers.0.bias:             -5.3e-01 /  1.1e-01 / -2.0e-01 / -1.9e-01
  layers.2.weight:           -1.5e+00 /  9.0e-01 / -4.1e-02 / -3.0e-02
  layers.2.bias:             -3.5e-01 /  2.9e-01 / -7.3e-02 / -6.7e-02
  final_linear.weight:       -6.4e-01 /  6.2e-01 /  7.5e-03 /  9.2e-03
  final_linear.bias:         -8.6e-02 /  1.4e-01 /  3.0e-02 /  3.4e-02
Model Critic min/max/mean/median:
  final_layers.0.weight:     -1.7e+00 /  8.9e-01 / -6.1e-02 / -4.2e-02
  final_layers.0.bias:       -3.2e-01 /  4.0e-01 / -3.3e-03 / -2.6e-02
  state_layers.0.weight:     -3.0e+00 /  3.3e+00 /  1.3e-02 /  1.3e-02
  state_layers.0.bias:       -7.1e-01 /  1.6e-01 / -2.5e-01 / -2.4e-01
  final_linear.weight:       -2.5e-01 /  2.7e-01 / -1.9e-02 / -2.7e-02
  final_linear.bias:          4.6e-01 /  4.6e-01 /  4.6e-01 /  4.6e-01

New policy noise: 0.1280
Episode  150  Score/Max/Avg: 17.6/30.2/14.6  AvStp: 1000  [μcL1/μcL2:  4.2e-03/ 4.1e-03 μaL: -4.3e-01] 
  Loaded steps:     150150
  Train iters:       74571
  Actor update:      37285
Model Actor min/max/mean/median:
  layers.0.weight:           -2.8e+00 /  2.7e+00 /  1.4e-02 /  1.3e-02
  layers.0.bias:             -5.4e-01 /  1.4e-01 / -2.3e-01 / -2.3e-01
  layers.2.weight:           -1.3e+00 /  9.5e-01 / -4.6e-02 / -3.1e-02
  layers.2.bias:             -5.1e-01 /  3.7e-01 / -1.0e-01 / -9.4e-02
  final_linear.weight:       -5.7e-01 /  7.0e-01 /  4.6e-04 /  6.8e-03
  final_linear.bias:         -9.5e-02 /  2.8e-01 /  7.5e-02 /  5.9e-02
Model Critic min/max/mean/median:
  final_layers.0.weight:     -2.4e+00 /  1.1e+00 / -7.1e-02 / -4.6e-02
  final_layers.0.bias:       -5.1e-01 /  4.7e-01 / -2.3e-02 / -4.5e-02
  state_layers.0.weight:     -3.9e+00 /  4.4e+00 /  1.2e-02 /  1.3e-02
  state_layers.0.bias:       -7.7e-01 /  2.9e-01 / -2.9e-01 / -2.9e-01
  final_linear.weight:       -2.1e-01 /  2.9e-01 / -2.7e-02 / -3.6e-02
  final_linear.bias:          1.0e+00 /  1.0e+00 /  1.0e+00 /  1.0e+00

New policy noise: 0.1024
Episode  200  Score/Max/Avg: 28.7/30.2/19.1  AvStp: 1000  [μcL1/μcL2:  9.6e-03/ 9.5e-03 μaL: -4.0e-01] 
  Loaded steps:     200200
  Train iters:       99600
  Actor update:      49800
Model Actor min/max/mean/median:
  layers.0.weight:           -3.3e+00 /  3.0e+00 /  9.2e-03 /  9.6e-03
  layers.0.bias:             -6.0e-01 /  1.5e-01 / -2.6e-01 / -2.8e-01
  layers.2.weight:           -1.6e+00 /  1.1e+00 / -4.7e-02 / -3.2e-02
  layers.2.bias:             -6.6e-01 /  4.2e-01 / -1.2e-01 / -1.2e-01
  final_linear.weight:       -7.6e-01 /  7.5e-01 /  1.0e-02 /  7.0e-03
  final_linear.bias:         -1.2e-01 /  3.5e-01 /  8.8e-02 /  6.1e-02
Model Critic min/max/mean/median:
  final_layers.0.weight:     -2.9e+00 /  1.4e+00 / -8.7e-02 / -5.4e-02
  final_layers.0.bias:       -6.5e-01 /  5.7e-01 / -2.0e-02 / -5.4e-02
  state_layers.0.weight:     -4.8e+00 /  5.7e+00 /  8.1e-03 /  8.3e-03
  state_layers.0.bias:       -8.6e-01 /  3.1e-01 / -3.2e-01 / -3.2e-01
  final_linear.weight:       -2.6e-01 /  3.5e-01 / -4.2e-02 / -4.9e-02
  final_linear.bias:          1.2e+00 /  1.2e+00 /  1.2e+00 /  1.2e+00

New policy noise: 0.0819
Episode  250  Score/Max/Avg: 27.3/37.6/23.4  AvStp: 1000  [μcL1/μcL2:  1.6e-02/ 1.6e-02 μaL: -8.7e-01] 
  Loaded steps:     250250
  Train iters:      124621
  Actor update:      62310
Model Actor min/max/mean/median:
  layers.0.weight:           -3.3e+00 /  3.4e+00 /  6.7e-03 /  9.1e-03
  layers.0.bias:             -6.6e-01 /  2.9e-01 / -2.8e-01 / -3.0e-01
  layers.2.weight:           -1.8e+00 /  1.5e+00 / -5.3e-02 / -3.8e-02
  layers.2.bias:             -6.1e-01 /  4.0e-01 / -1.5e-01 / -1.5e-01
  final_linear.weight:       -7.8e-01 /  7.1e-01 /  1.4e-02 /  1.3e-02
  final_linear.bias:         -1.3e-01 /  3.4e-01 /  1.1e-01 /  1.2e-01
Model Critic min/max/mean/median:
  final_layers.0.weight:     -3.5e+00 /  1.5e+00 / -1.0e-01 / -6.1e-02
  final_layers.0.bias:       -7.1e-01 /  8.2e-01 / -4.1e-02 / -8.1e-02
  state_layers.0.weight:     -5.2e+00 /  6.3e+00 /  9.0e-03 /  9.3e-03
  state_layers.0.bias:       -9.8e-01 /  3.5e-01 / -3.5e-01 / -3.5e-01
  final_linear.weight:       -2.8e-01 /  2.5e-01 / -5.8e-02 / -6.3e-02
  final_linear.bias:          1.7e+00 /  1.7e+00 /  1.7e+00 /  1.7e+00

New policy noise: 0.0655
Episode  300  Score/Max/Avg: 34.2/37.6/24.0  AvStp: 1000  [μcL1/μcL2:  3.9e-02/ 3.6e-02 μaL: -2.3e-01] 
  Loaded steps:     300300
  Train iters:      149650
  Actor update:      74825
Model Actor min/max/mean/median:
  layers.0.weight:           -3.6e+00 /  3.7e+00 /  6.6e-03 /  1.0e-02
  layers.0.bias:             -7.0e-01 /  3.4e-01 / -3.0e-01 / -3.1e-01
  layers.2.weight:           -2.0e+00 /  1.3e+00 / -5.6e-02 / -3.8e-02
  layers.2.bias:             -6.6e-01 /  3.6e-01 / -1.6e-01 / -1.4e-01
  final_linear.weight:       -7.2e-01 /  8.3e-01 /  1.1e-02 /  1.3e-02
  final_linear.bias:         -1.3e-01 /  3.1e-01 /  1.3e-01 /  1.8e-01
Model Critic min/max/mean/median:
  final_layers.0.weight:     -4.3e+00 /  1.9e+00 / -1.1e-01 / -6.3e-02
  final_layers.0.bias:       -7.8e-01 /  9.2e-01 / -1.2e-02 / -5.3e-02
  state_layers.0.weight:     -5.7e+00 /  7.5e+00 /  9.9e-03 /  1.1e-02
  state_layers.0.bias:       -1.1e+00 /  3.6e-01 / -3.8e-01 / -3.7e-01
  final_linear.weight:       -3.7e-01 /  3.1e-01 / -7.2e-02 / -7.6e-02
  final_linear.bias:          1.6e+00 /  1.6e+00 /  1.6e+00 /  1.6e+00

New policy noise: 0.0524
Episode  350  Score/Max/Avg: 25.3/37.6/18.8  AvStp: 1000  [μcL1/μcL2:  1.9e-01/ 1.9e-01 μaL:  4.6e+00] 
  Loaded steps:     350350
  Train iters:      174671
  Actor update:      87335
Model Actor min/max/mean/median:
  layers.0.weight:           -4.0e+00 /  3.9e+00 /  3.5e-03 /  8.8e-03
  layers.0.bias:             -8.2e-01 /  3.0e-01 / -3.3e-01 / -3.4e-01
  layers.2.weight:           -2.0e+00 /  1.3e+00 / -5.9e-02 / -4.1e-02
  layers.2.bias:             -7.7e-01 /  4.8e-01 / -1.8e-01 / -1.7e-01
  final_linear.weight:       -8.6e-01 /  8.5e-01 /  1.7e-02 /  8.0e-03
  final_linear.bias:         -9.8e-02 /  2.1e-01 /  6.2e-02 /  6.5e-02
Model Critic min/max/mean/median:
  final_layers.0.weight:     -4.5e+00 /  2.5e+00 / -1.2e-01 / -6.3e-02
  final_layers.0.bias:       -8.5e-01 /  1.0e+00 /  1.9e-02 / -4.4e-02
  state_layers.0.weight:     -6.3e+00 /  8.3e+00 /  3.0e-03 /  7.9e-03
  state_layers.0.bias:       -1.1e+00 /  6.0e-01 / -4.1e-01 / -4.1e-01
  final_linear.weight:       -5.8e-01 /  4.1e-01 / -1.0e-01 / -1.0e-01
  final_linear.bias:          1.1e+00 /  1.1e+00 /  1.1e+00 /  1.1e+00

New policy noise: 0.0500
Episode  400  Score/Max/Avg: 11.5/37.6/14.4  AvStp: 1000  [μcL1/μcL2:  1.2e-01/ 1.2e-01 μaL:  3.8e+00] 
  Loaded steps:     400400
  Train iters:      199700
  Actor update:      99850
Model Actor min/max/mean/median:
  layers.0.weight:           -4.4e+00 /  4.4e+00 /  3.4e-03 /  8.4e-03
  layers.0.bias:             -7.7e-01 /  3.1e-01 / -3.6e-01 / -3.8e-01
  layers.2.weight:           -2.2e+00 /  1.5e+00 / -6.0e-02 / -4.2e-02
  layers.2.bias:             -8.1e-01 /  5.4e-01 / -2.0e-01 / -1.8e-01
  final_linear.weight:       -9.4e-01 /  9.5e-01 /  5.9e-03 /  1.8e-03
  final_linear.bias:         -1.8e-01 /  2.8e-01 /  2.4e-02 / -1.7e-03
Model Critic min/max/mean/median:
  final_layers.0.weight:     -5.1e+00 /  3.3e+00 / -1.2e-01 / -6.7e-02
  final_layers.0.bias:       -9.0e-01 /  1.2e+00 /  1.2e-02 / -3.5e-02
  state_layers.0.weight:     -6.8e+00 /  8.9e+00 /  9.7e-04 /  8.6e-03
  state_layers.0.bias:       -1.2e+00 /  6.3e-01 / -4.4e-01 / -4.7e-01
  final_linear.weight:       -5.8e-01 /  3.5e-01 / -8.9e-02 / -7.7e-02
  final_linear.bias:          9.5e-01 /  9.5e-01 /  9.5e-01 /  9.5e-01
  
```
here is the training history:

![policy_noise_reduction](policy_noise_reduction.png)


