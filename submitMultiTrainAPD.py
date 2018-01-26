import os

# regularization terms to try
l = [10, 5, 1, 1e-1, 1e-2, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7]

# keep probabilites to try
keep_prob = [0.25, 0.5, 0.75, 1.0]

bodySL = """#!/bin/bash -l
#SBATCH -p regular
#SBATCH -t 19:00:00
#SBATCH -C haswell
#SBATCH -N 1
#SBATCH -L SCRATCH

cd /global/u2/m/maweber/EXODeep/projects/PosFromAPD/
%s
"""

bodySH = """#!/bin/bash -l
module load python
source activate deeplearning
python /global/u2/m/maweber/EXODeep/projects/PosFromAPD/TrainAPDPosRec.py --maxTrainSteps 50000 --batchSize 1000 --regTerm %.7f --dropout %.2f --outDir /global/u2/m/maweber/EXODeep/projects/PosFromAPD/output/RegTermDropoutStudy/%.7f_%.2f
"""

outDir = "/global/u2/m/maweber/EXODeep/projects/PosFromAPD/output/RegTermDropoutStudy"

for i in l:
	for k in keep_prob:
		out = "%s/%.7f_%.2f"%(outDir,i,k)
		cmd = 'mkdir %s'%out
		os.system(cmd)
		filenameSL = "%s/batchCori_%.7f_%.2f.sl"%(out,i,k)
		filenameSH = "%s/submitBatch_%.7f_%.2f.sh"%(out,i,k)
		file = open(filenameSL,'w')
		file.write(bodySL % filenameSH)
		file.close()
		file = open(filenameSH, 'w')
		file.write(bodySH % (i,k,i,k))
		file.close()
		cmd = 'chmod 755 %s'%filenameSH
		os.system(cmd)
		cmd = 'sbatch %s'%filenameSL
		print cmd
		os.system(cmd)

