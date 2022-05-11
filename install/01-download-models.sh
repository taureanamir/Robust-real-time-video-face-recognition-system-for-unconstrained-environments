#!/usr/bin/env bash

set -e

# 
PWD=`pwd`

echo -e "------------------------- \e[1;34m Current working dir: $PWD \e[0m ------------------------"

# 1. Download face and tracking models from QNAP

echo "--------------------------------------------------------------------------------"
echo -e "------------------------- \e[1;34m 1. Download face and tracking models from google drive. \e[0m ------------------------"
echo "--------------------------------------------------------------------------------"

pip3 install --upgrade pip
pip3 install gdown
gdown https://drive.google.com/uc?id=1XIOW5DVpkIMmdxc67L_CLuNyIc4A9XrS -O ../software/input/models.bz2

echo "--------------------------------------------------------------------------------"
echo -e "------------------------- \e[1;34m 2. Extract downloaded models. \e[0m ------------------------"
echo "--------------------------------------------------------------------------------"

tar -jxvf ../software/input/models.bz2 -C ../software/input


