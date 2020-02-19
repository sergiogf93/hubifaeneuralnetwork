#!/bin/bash

echo $PWD
#export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
#source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
#asetup 20.7.9.9.23,MCProd

cd /nfs/pic.es/user/s/sgonzalez/scratch2/DL/hubifaeneuralnetwork

source conda_setup.sh
conda activate base

python DeepNeuralNetwork.py -s ../ifaeneuralnetwork/Datasets/Znunu_TAR.pkl -b ../ifaeneuralnetwork/Datasets/Znunu_TAR.pkl -l BkgVsBkg_Randomizing -w --SigName Znunu --BkgName Znunu -x 1024 -e 200 --nodes 20 --depth 2 --lr 0.001 --train-size 0.8 --bkg-vs-bkg

