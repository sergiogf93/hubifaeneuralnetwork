#!/bin/bash

echo $PWD
#export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
#source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
#asetup 20.7.9.9.23,MCProd

cd /nfs/pic.es/user/s/sgonzalez/scratch2/DL/ifaeneuralnetwork

source conda_setup.sh
conda activate base

signal_file=${1}
bkg_file=${2}
label=${3}
signal_name=${4}
bkg_name=${5}
batch_size=${6}
epochs=${7}
fileOutput=${8}

python DeepNeuralNetwork.py -s ${signal_file} -b ${bkg_file} -l ${label} -w --SigName ${signal_name} --BkgName ${bkg_name} -x ${batch_size} -e ${epochs} -f ${fileOutput}

