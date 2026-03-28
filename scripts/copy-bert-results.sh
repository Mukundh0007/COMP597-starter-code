#!/bin/bash
# Copy BERT results from Slurm node storage to your repo (so you can open them without SSH).
# Run from repo root: ./scripts/copy-bert-results.sh

SCRIPTS_DIR=$(readlink -f -n $(dirname $0))
REPO_DIR=$(readlink -f -n ${SCRIPTS_DIR}/..)
SRC="/home/slurm/comp597/students/${USER}"
DEST="${REPO_DIR}"

"${SCRIPTS_DIR}/bash_srun.sh" "cp ${SRC}/bert_run_results.csv ${DEST}/ 2>/dev/null; cp ${SRC}/bert_run_plots.png ${DEST}/ 2>/dev/null; cp ${SRC}/bert_run_plots_util_throughput.png ${DEST}/ 2>/dev/null; echo 'Copied to ${DEST}'"
