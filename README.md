To get the code:

	lsetup git
	git clone https://:@gitlab.cern.ch:8443/gonzales/ifaeneuralnetwork.git

### IFAE NEURAL NETWORK FRAMEWORK

To run keras we first need to install anaconda and define and environment.

## CONDA INSTALLATION

Some of the at3 machines lack the libraries to do this so connecting to at301 is recommended. The procedure is as follows:

1. wget https://repo.continuum.io/archive/Anaconda2-2018.12-Linux-x86_64.sh
2. chmod +x Anaconda2-2018.12-Linux-x86_64.sh
3. ./Anaconda2-2018.12-Linux-x86_64.sh  

Now we can install what we need:

	pip install keras

Install numpy, pandas and any other package needed using pip.

To get tensorflow we use:

	conda install tensorflow

Which creates the environment "base" where we will use this framework. To enter and exit the environment use:

	conda deactivate

	conda activate base

## RUNNING THE CODE

To run the code you need to be in the environment where tensorflow was installed:

    conda activate base

We can get info on the parameters of the code with

	python DeepNeuralNetwork.py --help

An example of a basic command to run is

	python DeepNeuralNetwork.py -s Datasets/Whad_TAR.pkl -b Datasets/Znunu_TAR.pkl -l TEST -w --SigName Whad --BkgName Znunu --do-plots -f testFile
	
where:
 >   -s defines the pickle with the signal events
 
 >   -b defines the pickle with the bkg events
 
 >   -l sets the label for Plots, etc
 
 >   -w to train the NeuralNetwork taking the weights into account
 
 >   --SigName sets the label to use for the signal when plotting
 
 >   --BkgName sets the label to use for the background when plotting
 
 >   --doPlots to do the plots of the input parameters

This will create the plot of the discriminant in the Discriminants/ folder and will also store the signal to background ratio in testFile.dat

A command with more precise inputs would be:

    python DeepNeuralNetwork.py -s Datasets/Whad_TAR.pkl -b Datasets/Znunu_TAR.pkl -l TEST -w --SigName Whad --BkgName Znunu --doPlots -f testFile 
    -x 524 -e 100 --lr 0.01 --train-size 0.8

the new arguments are:
 >   -x set the batch size to 524 instead of the default 1024
 
 >   -e set the number of epochs to 100 instead of the default 200
 
 >   --lr set the learning rate to 0.01 instead of the default 0.001
 
 >   --train-size use 80% of the sample for training instead of 70% which is the default
    

## RUNNING IN CONDOR

Submitting jobs to CONDOR allows you to run in parallel multiple trainings. You can find more information about CONDOR here:

https://pwiki.pic.es/index.php?title=HTCondor_User_Guide
https://indico.cern.ch/event/828313/contributions/3466898/attachments/1869920/3076790/HTCondorUserGuidePIC.pdf

The command to submit jobs to CONDOR is:

    condor_submit condor_NN.sub
    
Inside condor_NN.sub the executable is defined as launch_condor.sh which you can open to check how DeepNeuralNetwork.py is executed.

The arguments must be in the submit_arguments.txt file. There will be a job submission per each line in that file. For example, if submit_arguments.txt contains the following lines:

    Datasets/Whad_TAR.pkl Datasets/Znunu_TAR.pkl monoW_vs_Znunu Whad Znunu 1000 200 Whad_vs_Znunu_StoB
    Datasets/Whad_TAR.pkl Datasets/Znunu_TAR.pkl monoW_vs_Znunu Whad Znunu 2500 200 Whad_vs_Znunu_StoB

It will launch the DeepNeuralNetwork using the Whad_TAR and Znunu_TAR samples but in one case it will use 1000 as batch size while in the other it will be 2500. The output will then stored to Whad_vs_Znunu_StoB.dat 

Submitting jobs to CONDOR is very useful when you want to study and optimize the parameters of the NN.

## RUN THE CONVOLUTIONAL NEURAL NETWORK (TO BE REVISED!!!!)

You can get the parameters of the code with

	python CNN.py --help

An example for running the code:

	python CNN.py -s imgFiles_ZqqZvv_monoJet -b imgFiles_Znunu_280_500CVetoBVeto -l _test

	python CNN.py -s invH -b Znunu -l _test

This line would go to the path /nfs/pic.es/user/s/sgonzalez/scratch2/DL/Jet_images/run/output/ and use imgFiles_ZqqZvv_monoJet as the signal folder of the images and imgFiles_Znunu_280_500CVetoBVeto as the bkg folder. The -l gives the label to the output plots, in this case, _test.

The channels for the images are divided as subfolders. For now we have pt and emf. The default is to use the pt channel.




python MesProves.py -s Datasets/Whad_TAR.pkl -b Datasets/Znunu_TAR.pkl -l TEST -w --SigName Whad --BkgName Znunu --do-plots -f nothing -x 524 -e 100 --lr 0.01 --train-size 0.8
