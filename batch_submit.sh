#!/bin/batch

filename=${1}
options=${2:-""}
SCRIPT="script_${filename}.sh"



echo "#!/bin/bash" > ${SCRIPT}

cat conda_setup.txt >> ${SCRIPT}


echo "cd \$PBS_O_WORKDIR" >> ${SCRIPT}

echo "conda activate base" >> ${SCRIPT}

echo "python DeepNeuralNetwork.py ${2} > out_${filename}.log  " >> ${SCRIPT}

qsub -q at3 ${SCRIPT}

