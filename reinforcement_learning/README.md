# Reinforcement Learning Pipeline

## Setup

Included in this folder are two files used for setup:
setup.sh and packages.txt. packages.txt includes a list of
all Python packages needed to run the code. Changing this
file may cause the code to be unable to run. To setup a
system for use, simply running setup.sh with:

```
chmod 0700 ./setup.sh
./setup.sh
```

is sufficient. It will require the input of
a sudo password, and pressing the enter key a few times to
configure everything. It is tested to work on an Ubuntu
18.04.5 system and will configure the system and create a
Python3.9 virtual environment for use.

*WARNING*: setup.sh will change the system's Python
installation from whatever is installed(Ubuntu 18.04.5
defaults to 3.6) to Python3.9. Please be wary of this.

## Training a Model

To run the training code, the procedure is straightforward.
The following two commands should be ran:

```
source venv/bin/activate
chmod 0700 ./rl_training.py
./rl_training.py
```

This will activate the Python3.9 virtual environment, then
run the training code to build the model. The code will save
the trained policy and create an evaluation video for
viewing the model's performance.

## Controlling the robot

To run a model that controls the robot through ROS, use the
`robot_controller.py` script. Specify the appropriate topics and the location of
the policy you want to use at the top of the script prior to running.
