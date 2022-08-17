#!/usr/bin/env bash

set -e
set -o xtrace
PWD_START=`pwd`

module purge
module load gcc/8.4.0-cuda
module load python/3.7.7
module load cuda/10.1
module load cmake

source ~/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda create -n ruslan python=3.6 cmake=3.14.0 -y
conda activate ruslan

cd ..
# ----------------------------------------------------------------------------
# install habitat-sim
# ----------------------------------------------------------------------------

conda install -y pytorch torchvision cudatoolkit=10.0 -c pytorch
conda clean -ya
rm -rf habitat-sim
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
git checkout 856d4b08c1a2632626bf0d205bf46471a99502b7 # v0.1.7
python -m pip install -r requirements.txt --prefix="$CONDA_PREFIX"

module purge
module load gcc/8.4.0
module load cuda/10.1
module load cmake
python setup.py install --headless --with-cuda --prefix="$CONDA_PREFIX"

cd ..
# ----------------------------------------------------------------------------
# install habitat-lab
# ----------------------------------------------------------------------------

rm -rf habitat-lab
git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
git checkout ac4339ed7e9bafb5e45fee9c5cf68095f40edee2 # challenge-2021

# install both habitat and habitat_baselines
pip install -r requirements.txt --prefix="$CONDA_PREFIX"
pip install -r habitat_baselines/rl/requirements.txt --prefix="$CONDA_PREFIX"
pip install -r habitat_baselines/rl/ddppo/requirements.txt --prefix="$CONDA_PREFIX"
pip install -r habitat_baselines/il/requirements.txt --prefix="$CONDA_PREFIX"
python setup.py develop --all --prefix="$CONDA_PREFIX"

cd ..
# ----------------------------------------------------------------------------
#   install pointgoal-navigation requirements
# ----------------------------------------------------------------------------
cd $PWD_START
python -m pip install -r requirements.txt --prefix="$CONDA_PREFIX"

# ----------------------------------------------------------------------------
#   download the dataset for Gibson PointNav
# ----------------------------------------------------------------------------
python -m pip install gdown --prefix="$CONDA_PREFIX"

gdown https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/gibson/v2/pointnav_gibson_v2.zip
mkdir -p data/datasets/pointnav/gibson/v2
unzip pointnav_gibson_v2.zip -d data/datasets/pointnav/gibson/v2
rm pointnav_gibson_v2.zip
gdown https://drive.google.com/uc?id=15_vh9rZgNhk_B8RFWZqmcW5JRdNQKM2G --output data/datasets/pointnav/gibson/gibson_quality_ratings.csv

#NOCOLOR='\033[0m'
#RED='\033[0;31m'
#echo -e "\n${RED}NOTE:${NOCOLOR} use command 'ln -s <path to scene datasets> ${PWD}/data/scene_datasets' to link the simulation scenes.\n"
