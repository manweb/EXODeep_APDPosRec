#!/bin/bash -l
module load python
source activate deeplearning
python TrainAPDPosRec.py --maxTrainStep 30000 --batchSize 500 --regTerm 1e-4 --learningRate 5e-4 --dropout 1.0 --outDir /global/u2/m/maweber/EXODeep/projects/PosFromAPD/output/position/083017/

