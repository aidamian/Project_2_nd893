# Continuous Control

This repository presents the [DDPG](https://arxiv.org/pdf/1509.02971.pdf)/[TD3](https://arxiv.org/pdf/1509.02971.pdf) implementaton of the solution for *Unity Reacher* environment.

## The environment

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.
The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

* The Reward 
  * The documentation states that +0.1 is provided for each time step that the agent's end effector is at the target locdation. However it seems that the actual reward is around 4e-2.
  
* The State space
  * 33 variables of 2 arms in total
  * position
  * rotation
  * velocity
  * angular velocities

* The Action space (Continuous)
  * Size of 4, corresponding to torque applicable to 2 arm joints
  * each of the _action_ variables is between -1 and +1
  
  
## Installation and requirements

This project is provided as a Jupyter-notebook-only source code. Here is the list of requirements for the server where the Jupyter must run:
- Python 3
- ML-Agents toolkit [https://github.com/Unity-Technologies/ml-agents]
- Unity Reacher environment [https://unity3d.com/machine-learning]
