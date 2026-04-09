#!/usr/bin/env bash
set -euo pipefail

SCRIPTS_DIR=$(readlink -f -n "$(dirname "$0")")
REPO_DIR=$(readlink -f -n "${SCRIPTS_DIR}/..")

# E2 uses the same batch/repeat pairs as E1.
BATCH_SIZES=(32 16 8)
RUNS_PER_BATCH=${E2_RUNS_PER_BATCH:-3}
COOLDOWN_S=${E2_COOLDOWN_S:-20}
TARGET_SECONDS=${E2_TARGET_SECONDS:-300}
BERT_SEED=${BERT_SEED:-42}
E1_SUMMARY_CSV="${REPO_DIR}/results/E1_step3/e1_time_summary.csv"

RESULTS_DIR="${REPO_DIR}/results/E2_step3"
LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

RUNS_CSV="${RESULTS_DIR}/e2_runs.csv"
SUMMARY_CSV="${RESULTS_DIR}/e2_time_overhead_summary.csv"

echo "timestamp,batch_size,repeat,run,wall_clock_s,train_time_s,steps,e1_avg_s,overhead_pct,within_5pct,codecarbon_dir,log_file" > "${RUNS_CSV}"
echo "batch_size,repeat,run1_s,run2_s,run3_s,avg_s,std_s,e1_avg_s,overhead_vs_e1_pct,within_5pct" > "${SUMMARY_CSV}"

first_run=true

run_e2_command() {
    local batch_size="$1"
    local repeat_val="$2"
    local run_num="$3"
    local output_dir="$4"
    local project_name="$5"

    "${SCRIPTS_DIR}/srun.sh" \
        --logging.level INFO \
        --model bert \
        --data bert \
        --trainer simple \
        --batch_size "${batch_size}" \
        --learning_rate 1e-4 \
        --trainer_stats codecarbon_v2 \
        --trainer_stats_configs.codecarbon_v2.run_num "${run_num}" \
        --trainer_stats_configs.codecarbon_v2.project_name "${project_name}" \
        --trainer_stats_configs.codecarbon_v2.output_dir "${output_dir}" \
        --trainer_stats_configs.codecarbon_v2.sync_every_steps 3 \
        --trainer_stats_configs.codecarbon_v2.measure_power_secs 0.5 \
        --data_configs.bert.seed "${BERT_SEED}" \
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

get_e1_avg_for_batch() {
    local batch_size="$1"
    if [[ ! -f "${E1_SUMMARY_CSV}" ]]; then
        echo ""
        return
    fi
    awk -F',' -v b="${batch_size}" 'NR > 1 && $1 == b { print $6 }' "${E1_SUMMARY_CSV}" | tail -1
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

calc_overhead_pct() {
    local measured_s="$1"
    local baseline_s="$2"
    if [[ -z "${baseline_s}" ]]; then
        echo ""
        return
    fi
    awk -v m="${measured_s}" -v b="${baseline_s}" 'BEGIN {
        if (b <= 0) { print ""; exit }
        printf "%.4f", ((m - b) / b) * 100.0
    }'
}

is_within_5pct() {
    local overhead_pct="$1"
    if [[ -z "${overhead_pct}" ]]; then
        echo ""
        return
    fi
    awk -v o="${overhead_pct}" 'BEGIN {
        ao = (o < 0) ? -o : o;
        if (ao <= 5.0) print "yes"; else print "no"
    }'
}

for batch_size in "${BATCH_SIZES[@]}"; do
    repeat_val=$(get_repeat_for_batch "${batch_size}")
    e1_avg_s=$(get_e1_avg_for_batch "${batch_size}")
    run1_s=""
    run2_s=""
    run3_s=""

    for run in $(seq 1 "${RUNS_PER_BATCH}"); do
        if [[ "${first_run}" == "false" ]]; then
            echo "[E2] Cooling down for ${COOLDOWN_S}s before next run..."
            sleep "${COOLDOWN_S}"
        fi
        first_run=false

        run_id="batch${batch_size}_run${run}"
        run_output_dir="${RESULTS_DIR}/${run_id}"
        run_codecarbon_dir="${run_output_dir}/codecarbon"
        mkdir -p "${run_output_dir}" "${run_codecarbon_dir}"

        log_file="${LOG_DIR}/${run_id}.log"
        run_csv="${run_output_dir}/run_metadata.csv"
        project_name="bert-e2-bs${batch_size}"

        echo "[E2] Starting ${run_id} (batch_size=${batch_size}, repeat=${repeat_val}, trainer_stats=codecarbon_v2)"

        start_ns=$(date +%s%N)
        {
            echo "[E2] Command start: $(date -Iseconds)"
        echo "[E2] Command: ${SCRIPTS_DIR}/srun.sh --logging.level INFO --model bert --data bert --trainer simple --batch_size ${batch_size} --learning_rate 1e-4 --trainer_stats codecarbon_v2 --trainer_stats_configs.codecarbon_v2.run_num ${run} --trainer_stats_configs.codecarbon_v2.project_name ${project_name} --trainer_stats_configs.codecarbon_v2.output_dir ${run_codecarbon_dir} --trainer_stats_configs.codecarbon_v2.sync_every_steps 5 --trainer_stats_configs.codecarbon_v2.measure_power_secs 0.5 --data_configs.bert.repeat ${repeat_val} --data_configs.bert.n 0"
            run_e2_command "${batch_size}" "${repeat_val}" "${run}" "${run_codecarbon_dir}" "${project_name}"
            echo "[E2] Command end: $(date -Iseconds)"
        } 2>&1 | tee "${log_file}"
        end_ns=$(date +%s%N)

        wall_clock_s=$(awk -v s="${start_ns}" -v e="${end_ns}" 'BEGIN { printf "%.4f", (e-s)/1000000000.0 }')
        train_time_s=$(grep -Eo '\[BERT\] Total training time: [0-9]+\.[0-9]+' "${log_file}" | tail -1 | awk '{print $5}' || true)
        steps=$(grep -Eo 'over [0-9]+ steps\.' "${log_file}" | tail -1 | awk '{print $2}' || true)

        if [[ -z "${train_time_s}" ]]; then
            train_time_s="${wall_clock_s}"
        fi
        if [[ -z "${steps}" ]]; then
            steps=""
        fi

        overhead_pct=$(calc_overhead_pct "${train_time_s}" "${e1_avg_s}")
        within_5pct=""
        if [[ -n "${overhead_pct}" ]]; then
            within_5pct=$(is_within_5pct "${overhead_pct}")
        fi

        {
            echo "batch_size,repeat,run,trainer_stats,train_time_s,wall_clock_s,steps,target_seconds,e1_avg_s,overhead_pct,within_5pct,codecarbon_output_dir"
            echo "${batch_size},${repeat_val},${run},codecarbon_v2,${train_time_s},${wall_clock_s},${steps},${TARGET_SECONDS},${e1_avg_s},${overhead_pct},${within_5pct},${run_codecarbon_dir}"
        } > "${run_csv}"

        ts=$(date -Iseconds)
        echo "${ts},${batch_size},${repeat_val},${run},${wall_clock_s},${train_time_s},${steps},${e1_avg_s},${overhead_pct},${within_5pct},${run_codecarbon_dir},${log_file}" >> "${RUNS_CSV}"

        if [[ "${run}" == "1" ]]; then run1_s="${train_time_s}"; fi
        if [[ "${run}" == "2" ]]; then run2_s="${train_time_s}"; fi
        if [[ "${run}" == "3" ]]; then run3_s="${train_time_s}"; fi

        echo "[E2] Finished ${run_id}: train_time_s=${train_time_s}, overhead_pct=${overhead_pct:-N/A}, within_5pct=${within_5pct:-N/A}"
    done

    if [[ "${RUNS_PER_BATCH}" -ne 3 ]]; then
        avg_s=""
        std_s=""
        avg_overhead_pct=""
        avg_within_5pct=""
    else
        avg_s=$(calc_mean "${run1_s}" "${run2_s}" "${run3_s}")
        std_s=$(calc_std "${run1_s}" "${run2_s}" "${run3_s}")
        avg_overhead_pct=$(calc_overhead_pct "${avg_s}" "${e1_avg_s}")
        avg_within_5pct=""
        if [[ -n "${avg_overhead_pct}" ]]; then
            avg_within_5pct=$(is_within_5pct "${avg_overhead_pct}")
        fi
    fi

    echo "${batch_size},${repeat_val},${run1_s},${run2_s},${run3_s},${avg_s},${std_s},${e1_avg_s},${avg_overhead_pct},${avg_within_5pct}" >> "${SUMMARY_CSV}"
    echo "[E2] Batch ${batch_size} summary: avg_s=${avg_s:-N/A}, overhead_vs_e1_pct=${avg_overhead_pct:-N/A}, within_5pct=${avg_within_5pct:-N/A}"
done

echo "[E2] Done. Artifacts:"
echo "  - ${SUMMARY_CSV}"
echo "  - ${RUNS_CSV}"
echo "  - ${RESULTS_DIR}/batch*_run*/run_metadata.csv"
echo "  - ${RESULTS_DIR}/batch*_run*/codecarbon/*"
echo "  - ${LOG_DIR}/batch*_run*.log"
