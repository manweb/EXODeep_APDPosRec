#!/bin/bash -l
module load python/2.7-anaconda-4.4
python TrainCNNAPDPosRec.py --maxTrainStep 30000 --batchSize 256 --regTerm 5e-3 --learningRate 2e-6 --dropout 1.0 --model /global/cscratch1/sd/maweber/output/CNN/position/121417/training1/model_0.0050000_0.0000020-17000.meta --outDir /global/cscratch1/sd/maweber/output/CNN/position/121417/training1/ --trainingSet /global/cscratch1/sd/maweber/APDWFSignalsEnergyThreshold_Flattened_Merged_Train_121417.csv --testSet /global/cscratch1/sd/maweber/APDWFSignalsEnergyThreshold_Flattened_Merged_Test_121417.csv

