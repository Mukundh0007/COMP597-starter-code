#!/bin/bash
# BERT with synthetic data (milabench-style). Default: loss + timing + plots.
# For GPU power/energy: use --trainer_stats codecarbon and codecarbon configs (see below).

SCRIPTS_DIR=$(readlink -f -n $(dirname $0))

${SCRIPTS_DIR}/srun.sh \
    --logging.level INFO \
    --model bert \
    --data bert \
    --trainer simple \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --trainer_stats simple \
    --data_configs.bert.repeat 750 \
    --data_configs.bert.n 0

### BERT with CodeCarbon (power/energy for workload evaluation)
# ${SCRIPTS_DIR}/srun.sh \
#     --logging.level INFO \
#     --model bert \
#     --data bert \
#     --trainer simple \
#     --batch_size 4 \
#     --learning_rate 1e-4 \
#     --trainer_stats codecarbon \
#     --trainer_stats_configs.codecarbon.run_num 1 \
#     --trainer_stats_configs.codecarbon.project_name bert-synth \
#     --trainer_stats_configs.codecarbon.output_dir '${COMP597_JOB_STUDENT_STORAGE_DIR}/bert-codecarbon'