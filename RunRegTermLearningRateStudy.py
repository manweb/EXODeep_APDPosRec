import os

reg = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
lr = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 4e-4]

outDir = '/global/u2/m/maweber/EXODeep/projects/PosFromAPD/output/RegTermLearningRateStudy/'

for i in reg:
	for k in lr:
		cmd = 'python TrainAPDPosRec.py --maxTrainSteps 1000 --batchSize 1000 --regTerm %f --learningRate %f --outDir %s'%(i,k,outDir)
		os.system(cmd)

