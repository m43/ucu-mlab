#!/usr/bin/env bash

N_CORES=16

set -e
set -o xtrace
PWD_START=`pwd`

module purge
module load gcc/8.4.0-cuda
module load python/3.7.7
module load cuda/10.1
module load cmake
# module load NCCL/2.4.8-1-cuda.10.0

rm -rf ./venv
python3 -m venv ./venv
source ./venv/bin/activate
pip install --upgrade pip

cd ..

#CMAKE_INSTALL_PREFIX="$HOME"/cmake_3.14
#mkdir -p "$CMAKE_INSTALL_PREFIX"
#wget https://cmake.org/files/v3.14/cmake-3.14.7-Linux-x86_64.sh
#bash cmake-3.14.7-Linux-x86_64.sh --skip-license --prefix="$CMAKE_INSTALL_PREFIX"
#rm cmake-3.14.7-Linux-x86_64.sh
#PATH_BKP="$PATH"
#export PATH="$CMAKE_INSTALL_PREFIX/bin:$PATH"
#cmake --version

#GCC_7_3_0_INSTALL_PREFIX="$HOME/gcc-7.3.0"
#mkdir -p "$GCC_7_3_0_INSTALL_PREFIX"
#wget https://ftp.gwdg.de/pub/misc/gcc/releases/gcc-7.3.0/gcc-7.3.0.tar.gz
#tar xf gcc-7.3.0.tar.gz
#rm gcc-7.3.0.tar.gz
#cd gcc-7.3.0
#./configure --disable-multilib -prefix="$GCC_7_3_0_INSTALL_PREFIX"
#make -jCORES $N_CORES
#make install
#PATH_BKP="$PATH"
#export PATH="$GCC_7_3_0_INSTALL_PREFIX/bin:$PATH"
#gcc --version

# ----------------------------------------------------------------------------
# install habitat-sim
# ----------------------------------------------------------------------------

pip install torchvision torch
# cudatoolkit=10.0

rm -rf habitat-sim
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
git checkout 856d4b08c1a2632626bf0d205bf46471a99502b7 # v0.1.7
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
git checkout ac4339ed7e9bafb5e45fee9c5cf68095f40edee2 # challenge-2021

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
pip install torch==1.7.1 torchvision==0.8.2 segmentation-models-pytorch==0.1.3

# ----------------------------------------------------------------------------
#   download the dataset for Gibson PointNav
# ----------------------------------------------------------------------------
pip install gdown

gdown https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/gibson/v2/pointnav_gibson_v2.zip
mkdir -p data/datasets/pointnav/gibson/v2
unzip pointnav_gibson_v2.zip -d data/datasets/pointnav/gibson/v2
rm pointnav_gibson_v2.zip
gdown https://drive.google.com/uc?id=15_vh9rZgNhk_B8RFWZqmcW5JRdNQKM2G --output data/datasets/pointnav/gibson/gibson_quality_ratings.csv

ln -s ~/data data/scene_datasets