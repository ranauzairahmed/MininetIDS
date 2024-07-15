#!/bin/bash

# Update package list
sudo apt-get update

# Install Mininet
git clone https://github.com/mininet/mininet
cd mininet
./util/install.sh -a
cd ..

# Fix any missing dependencies (optional)
sudo apt-get update --fix-missing

# Install Python3-pip
sudo apt install -y python3-pip

# Install Ryu Controller
git clone https://github.com/osrg/ryu.git
cd ryu
sudo python3 ./setup.py install
sudo pip3 install --upgrade ryu
cd ..

# Install additional requirements
pip3 install -r requirements.txt

echo "Installation completed!"
