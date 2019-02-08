# Robot-Localization-with-Particle-Filter-and-Extend-Kalman-Filter
# filters

The starter code is written in Python and depends on NumPy and Matplotlib.
This README gives a brief overview of each file.

- `localization.py` -- This is the main entry point for running experiments.
- `soccer_field.py` -- This implements the dynamics and observation functions,
  as well as the noise models for both.
- `utils.py` -- This contains assorted plotting functions, as well as a useful
  function for normalizing angles.
- `policies.py` -- This contains a simple policy, which you can safely ignore.
- `ekf.py` -- Kalman filter implementation
- `pf.py` -- Particle filter implementation

## Command-Line Interface

To visualize the robot in the soccer field environment, run
```bash
$ python localization.py --plot none
```
The blue line traces out the robot's position, which is a result of noisy actions.
The green line traces the robot's position assuming that actions weren't noisy.

After you implement a filter, the filter's estimate of the robot's position will be drawn in red.
```bash
$ python localization.py --plot ekf
$ python localization.py --plot pf
```

You can scale the noise factors for the data generation process or the filters
with the `--data-factor` and `--filter-factor` flags. To see other command-line
flags available to you, run
```bash
$ python localization.py -h
```
## Result
You can input this code to check the result of EKF:
```bash
$ python localization.py ekf --seed 0
```
Mean position error: 8.998367536084695
Mean Mahalanobis error: 4.416418248584291
ANEES: 1.4721394161947636

You can input this code to check the result of PF:
```bash
$ python localization.py pf --seed 0
```
Mean position error: 8.567264372950907
Mean Mahalanobis error: 14.742252771106521
ANEES: 4.914084257035507

## Visualization
![ekf](https://user-images.githubusercontent.com/36937088/52497244-703c6c00-2b8a-11e9-9da0-736e13511655.jpeg)
![pf](https://user-images.githubusercontent.com/36937088/52497360-bdb8d900-2b8a-11e9-964a-8e27c397654b.jpeg)
