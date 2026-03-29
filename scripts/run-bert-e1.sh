#!/usr/bin/env bash
set -euo pipefail

SCRIPTS_DIR=$(readlink -f -n "$(dirname "$0")")
REPO_DIR=$(readlink -f -n "${SCRIPTS_DIR}/..")

# E1 protocol defaults.
# Fixed pairs provided by user for approximately 5-minute runs.
BATCH_SIZES=(32 16 8)
RUNS_PER_BATCH=${E1_RUNS_PER_BATCH:-3}
COOLDOWN_S=${E1_COOLDOWN_S:-60}

RESULTS_DIR="${REPO_DIR}/results/E1"
LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

SUMMARY_CSV="${RESULTS_DIR}/e1_time_summary.csv"
RUNS_CSV="${RESULTS_DIR}/e1_runs.csv"

echo "batch_size,repeat,run1_s,run2_s,run3_s,avg_s,std_s" > "${SUMMARY_CSV}"
echo "timestamp,batch_size,repeat,run,wall_clock_s,train_time_s,steps,run_csv,log_file" > "${RUNS_CSV}"

first_run=true

run_e1_command() {
    local batch_size="$1"
    local repeat_val="$2"
    "${SCRIPTS_DIR}/srun.sh" \
        --logging.level INFO \
        --model bert \
        --data bert \
        --trainer simple \
        --batch_size "${batch_size}" \
        --learning_rate 1e-4 \
        --trainer_stats noop \
        --data_configs.bert.repeat "${repeat_val}" \
        --data_configs.bert.n 0
}

get_repeat_for_batch() {
    local batch_size="$1"
    case "${batch_size}" in
        32) echo "550" ;;
        16) echo "1100" ;;
        8) echo "2000" ;;
        *)
            echo "Unknown batch size '${batch_size}'. Supported values are 32, 16, 8." >&2
            exit 1
            ;;
    esac
}

calc_mean() {
    awk -v a="$1" -v b="$2" -v c="$3" 'BEGIN { printf "%.4f", (a+b+c)/3.0 }'
}

calc_std() {
    awk -v a="$1" -v b="$2" -v c="$3" 'BEGIN {
        m=(a+b+c)/3.0;
        v=((a-m)*(a-m) + (b-m)*(b-m) + (c-m)*(c-m))/3.0;
        if (v < 0) v = 0;
        printf "%.4f", sqrt(v)
    }'
}

for batch_size in "${BATCH_SIZES[@]}"; do
    repeat_val=$(get_repeat_for_batch "${batch_size}")
    run1_s=""
    run2_s=""
    run3_s=""

    for run in $(seq 1 "${RUNS_PER_BATCH}"); do
        if [[ "${first_run}" == "false" ]]; then
            echo "[E1] Cooling down for ${COOLDOWN_S}s before next run..."
            sleep "${COOLDOWN_S}"
        fi
        first_run=false

        run_id="batch${batch_size}_run${run}"
        log_file="${LOG_DIR}/${run_id}.log"
        run_csv="${RESULTS_DIR}/results_${run_id}.csv"

        echo "[E1] Starting ${run_id} (batch_size=${batch_size}, repeat=${repeat_val}, trainer_stats=noop)"

        start_ns=$(date +%s%N)

        # Capture full output for reproducibility and post-run parsing.
        {
            echo "[E1] Command start: $(date -Iseconds)"
            echo "[E1] Command: ${SCRIPTS_DIR}/srun.sh --logging.level INFO --model bert --data bert --trainer simple --batch_size ${batch_size} --learning_rate 1e-4 --trainer_stats noop --data_configs.bert.repeat ${repeat_val} --data_configs.bert.n 0"
            run_e1_command "${batch_size}" "${repeat_val}"
            echo "[E1] Command end: $(date -Iseconds)"
        } 2>&1 | tee "${log_file}"

        end_ns=$(date +%s%N)
        wall_clock_s=$(awk -v s="${start_ns}" -v e="${end_ns}" 'BEGIN { printf "%.4f", (e-s)/1000000000.0 }')

        # Parse the trainer-reported runtime and steps when available.
        train_time_s=$(grep -Eo '\[BERT\] Total training time: [0-9]+\.[0-9]+' "${log_file}" | tail -1 | awk '{print $5}' || true)
        steps=$(grep -Eo 'over [0-9]+ steps\.' "${log_file}" | tail -1 | awk '{print $2}' || true)

        if [[ -z "${train_time_s}" ]]; then
            train_time_s="${wall_clock_s}"
        fi
        if [[ -z "${steps}" ]]; then
            steps=""
        fi

        # E1 is no-op stats by protocol: persist run metadata rather than per-step instrumentation.
        {
            echo "batch_size,repeat,run,trainer_stats,train_time_s,wall_clock_s,steps,note"
            echo "${batch_size},${repeat_val},${run},noop,${train_time_s},${wall_clock_s},${steps},E1 baseline run without per-step instrumentation"
        } > "${run_csv}"

        ts=$(date -Iseconds)
        echo "${ts},${batch_size},${repeat_val},${run},${wall_clock_s},${train_time_s},${steps},${run_csv},${log_file}" >> "${RUNS_CSV}"

        if [[ "${run}" == "1" ]]; then run1_s="${train_time_s}"; fi
        if [[ "${run}" == "2" ]]; then run2_s="${train_time_s}"; fi
        if [[ "${run}" == "3" ]]; then run3_s="${train_time_s}"; fi

        echo "[E1] Finished ${run_id}: train_time_s=${train_time_s}, wall_clock_s=${wall_clock_s}, steps=${steps}"
    done

    if [[ "${RUNS_PER_BATCH}" -ne 3 ]]; then
        echo "[E1] RUNS_PER_BATCH=${RUNS_PER_BATCH}; summary csv expects 3 runs. Filling missing values with empty fields."
        avg_s=""
        std_s=""
    else
        avg_s=$(calc_mean "${run1_s}" "${run2_s}" "${run3_s}")
        std_s=$(calc_std "${run1_s}" "${run2_s}" "${run3_s}")
    fi

    echo "${batch_size},${repeat_val},${run1_s},${run2_s},${run3_s},${avg_s},${std_s}" >> "${SUMMARY_CSV}"
    echo "[E1] Batch size ${batch_size} summary: avg_s=${avg_s:-N/A}, std_s=${std_s:-N/A}"
done

echo "[E1] Done. Artifacts:"
echo "  - ${SUMMARY_CSV}"
echo "  - ${RUNS_CSV}"
echo "  - ${RESULTS_DIR}/results_batch*_run*.csv"
echo "  - ${LOG_DIR}/batch*_run*.log"
