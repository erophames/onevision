#!/bin/bash

sudo apt-get update
sudo apt-get install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-dev python3.10-distutils
sudo apt-get install -y python3.10-venv
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
sudo python3.10 get-pip.py
rm get-pip.py
python3.10 -m pip install --upgrade virtualenv
python3.10 -m venv pathogen_detector_venv
source pathogen_detector_venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup Complete."
