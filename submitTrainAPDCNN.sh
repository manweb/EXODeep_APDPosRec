#!/bin/bash -l
module load python/2.7-anaconda-4.4
python TrainCNNAPDPosRec.py --maxTrainStep 9000 --batchSize 256 --regTerm 1e-4 --learningRate 5e-4 --dropout 1.0 --model /global/cscratch1/sd/maweber/output/CNN/111217/model_0.0001000_1.00-3000.meta --outDir /global/cscratch1/sd/maweber/output/CNN/111217/ --trainEnergy

