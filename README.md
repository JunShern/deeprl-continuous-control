# deeprl-continuous-control
This repository contains my submission for Project 2: Continuous Control of the [Udacity Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

## Project Details

The assignment is to train an agent that solves the Unity ML-Agents [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

The solution implements a Deep Deterministic Policy Gradient algorithm based on [[1]](#ddpg_paper) to solve the environment. For implementation and algorithm details, please see [Report.md](Report.md).

![trained_agent](assets/early_agent.gif)

_The goal of the environment is to effectively control a double-jointed robot arm to follow a moving target location (visualized as a green sphere). The environment provides the option of parallel agents for more effective training; the gif above shows 20 such agents performing the task at an early stage of training - some do well, some do poorly. To see all 20 agents performing well, check out [Report.md](Report.md)!_

#### Environment

_(The below description is replicated from the [udacity/deep-reinforcement-learning](https://github.com/udacity/deep-reinforcement-learning/blob/master/p1_navigation/README.md) repository.)_

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

**Distributed Training**

For this project, we will provide you with two separate versions of the Unity environment:
- The first version contains a single agent.
- The second version contains 20 identical agents, each with its own copy of the environment.

#### Solve criteria
The project assignment provides two options for solving the environment, corresponding to the single-agent environment and the 20-agent environment:

- **Option 1:** Solve the First Version The task is episodic, and in order to solve the environment, your agent must get an average score of +30 over 100 consecutive episodes. 
- **Option 2:** Solve the Second Version The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents. In particular, your agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).

## Getting started

#### Prerequisites
- Python >= 3.6
- A GPU is recommended but not required; however training on CPU may take up to 20 hours to solve the environment

#### Installation
1. Clone the repository.
```bash
git clone https://github.com/JunShern/deeprl-continuous-control.git
```

2. Create a virtual environment to manage your dependencies.
```bash
cd deeprl-continuous-control/
python3 -m venv .venv
source .venv/bin/activate # Activate the virtualenv
```

3. Install python dependencies
```bash
cd deeprl-continuous-control/python
pip install .
```

4. Download the Reacher environment from one of the links below:
    - Single Agent: [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip) | [Linux Headless](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) | [Max OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip) | [Windows 32-bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip) | [Windows 64-bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)
    - 20 Agents: [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip) | [Linux Headless](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) | [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip) | [Windows 32-bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip) | [Windows 64-bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
  
    _(For training on a remote server without virtual display e.g. AWS, use "Linux Headless")_

    Place the file in the `./env` folder of this repository, and unzip (or decompress) the file.

## Instructions

There are two entrypoints for the project: `train.py` to train the model from scratch, and `run.py` to run a trained agent in the environment.

1. Activate the virtualenv
```bash
cd deeprl-continuous-control/
source .venv/bin/activate # Activate the virtualenv
```
2. Run training. (Skip if you just want to run using the solved models in `./models`) 
```bash
python train.py
``` 
3. To run a trained agent.
```bash
python run.py
```
4. Exit the virtualenv when done
```bash
deactivate
```

## References

- <a name="ddpg_paper">[1]</a> Lillicrap, Timothy P., et al. "Continuous control with deep reinforcement learning." arXiv preprint arXiv:1509.02971 (2015).