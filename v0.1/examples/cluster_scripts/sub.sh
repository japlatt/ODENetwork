#!/bin/bash
#$ -t 1-6
#$ -N antennal_lobe
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -m beas
#$ -o ./output
#$ -e ./error
#$ -q batch.q
source activate python364
python run_script.py $SGE_TASK_ID
