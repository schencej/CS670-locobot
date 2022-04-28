#!/bin/bash

sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.9 -y
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 2
sudo update-alternatives --config python3
python3 -V
sudo apt install python3.9-distutils
python3 -m pip install --upgrade pip
sudo apt install python3.9-venv libpython3.9 -y

mkdir rl_venv
python3 -m venv rl_venv
sudo apt-get install -y xvfb ffmpeg freeglut3-dev
source rl_venv/bin/activate
pip3 install -r packages.txt
