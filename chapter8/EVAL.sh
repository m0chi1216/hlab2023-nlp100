#!/bin/bash

#PJM -L rscgrp=cx-share
#PJM -L gpu=1
#PJM -L elapse=1:00:00
#PJM -L jobenv=singularity
#PJM -j

eval "$(/home/z40351r/anaconda3/bin/conda shell.bash hook)"
conda activate nlp

python3 evaluation.py \