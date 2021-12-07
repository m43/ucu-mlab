#!/usr/bin/env bash

set -e
set -o xtrace
PWD_START=`pwd`

module purge
module load gcc/8.4.0-cuda
module load python/3.7.7
module load cuda/10.1
module load cmake

rm -rf ./venv_update
python3 -m venv ./venv_update
source ./venv_update/bin/activate
pip install --upgrade pip

cd ..

# ----------------------------------------------------------------------------
# install habitat-sim
# ----------------------------------------------------------------------------

pip install torchvision torch
# cudatoolkit=10.0

rm -rf habitat-sim
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
#git checkout 856d4b08c1a2632626bf0d205bf46471a99502b7 # v0.1.7
pip install -r requirements.txt

module purge
module load gcc/8.4.0
module load cuda/10.1
#module load cudnn/7.4
module load cmake
python setup.py install --headless --with-cuda --prefix="$VIRTUAL_ENV"

# silence habitat-sim logs
export GLOG_minloglevel=2
export MAGNUM_LOG="quiet"

cd ..

# ----------------------------------------------------------------------------
# install habitat-lab
# ----------------------------------------------------------------------------

rm -rf habitat-lab
git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
#git checkout ac4339ed7e9bafb5e45fee9c5cf68095f40edee2 # challenge-2021

# install both habitat and habitat_baselines
pip install -r requirements.txt
pip install -r habitat_baselines/rl/requirements.txt
pip install -r habitat_baselines/rl/ddppo/requirements.txt
pip install -r habitat_baselines/il/requirements.txt
python setup.py develop --all

cd ..

# ----------------------------------------------------------------------------
#   install pointgoal-navigation requirements
# ----------------------------------------------------------------------------

cd $PWD_START
#pip install torch==1.7.1 torchvision==0.8.2 segmentation-models-pytorch==0.1.3
pip install torch torchvision segmentation-models-pytorch

# ----------------------------------------------------------------------------
#   download the dataset for Gibson PointNav
# ----------------------------------------------------------------------------
#pip install gdown
#
#gdown https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/gibson/v2/pointnav_gibson_v2.zip
#mkdir -p data/datasets/pointnav/gibson/v2
#unzip pointnav_gibson_v2.zip -d data/datasets/pointnav/gibson/v2
#rm pointnav_gibson_v2.zip
#gdown https://drive.google.com/uc?id=15_vh9rZgNhk_B8RFWZqmcW5JRdNQKM2G --output data/datasets/pointnav/gibson/gibson_quality_ratings.csv
#
#ln -s ~/data data/scene_datasets