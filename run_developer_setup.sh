#!/bin/bash 

# Install dependency packages
pip install -r requirements.txt

# Install irharn in developer mode
python setup.py develop
#pip install -e .
