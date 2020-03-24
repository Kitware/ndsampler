#!/bin/bash 

# Install dependency packages
pip install -r requirements.txt

# Install in developer mode
#python setup.py develop
pip install -e .
