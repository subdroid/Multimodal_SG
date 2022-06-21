#!/bin/bash
qsub -q 'gpu*' -l gpu=8,gpu_ram=4G -pe smp 10 ./ft_gpt_universal.sh "gpt_base"
qsub -q 'gpu*' -l gpu=8,gpu_ram=8G -pe smp 10 ./ft_gpt_universal.sh "gpt_medium"
qsub -q 'gpu*' -l gpu=8,gpu_ram=16G -pe smp 10 ./ft_gpt_universal.sh "gpt_large"