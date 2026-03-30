#!/usr/bin/env bash
set -euo pipefail

SCRIPTS_DIR=$(readlink -f -n "$(dirname "$0")")
REPO_DIR=$(readlink -f -n "${SCRIPTS_DIR}/..")

# E3 protocol defaults.
BATCH_SIZES=(32 16 8)
RUNS_PER_BATCH=${E3_RUNS_PER_BATCH:-3}
COOLDOWN_S=${E3_COOLDOWN_S:-60}
TARGET_SECONDS=${E3_TARGET_SECONDS:-300}

RESULTS_DIR="${REPO_DIR}/results/E3"
LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

RUNS_CSV="${RESULTS_DIR}/e3_runs.csv"
TIME_SUMMARY_CSV="${RESULTS_DIR}/e3_time_summary.csv"

echo "timestamp,batch_size,repeat,run,wall_clock_s,train_time_s,steps,results_csv,timeline_csv,run_dir,log_file" > "${RUNS_CSV}"
echo "batch_size,repeat,run1_s,run2_s,run3_s,avg_s,std_s,target_seconds" > "${TIME_SUMMARY_CSV}"

first_run=true

run_e3_command() {
    local batch_size="$1"
    local repeat_val="$2"
    local output_dir="$3"

    BERT_STATS_OUTPUT_DIR="${output_dir}" "${SCRIPTS_DIR}/srun.sh" \
        --logging.level INFO \
        --model bert \
        --data bert \
        --trainer simple \
        --batch_size "${batch_size}" \
        --learning_rate 1e-4 \
        --trainer_stats simple \
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

aggregate_phase_summary() {
    local batch_size="$1"
    local out_csv="$2"
    local files=()

    while IFS= read -r f; do
        files+=("$f")
    done < <(find "${RESULTS_DIR}" -type f -path "*/batch${batch_size}_run*/metrics/bert_run_results.csv" | sort)

    if [[ ${#files[@]} -eq 0 ]]; then
        echo "No bert_run_results.csv found for batch ${batch_size}" >&2
        exit 1
    fi

    awk -F',' -v BATCH="${batch_size}" '
        FNR == 1 { next }
        {
            n += 1
            step_sum += $3; step_sq += ($3 * $3)
            f_sum += $4; f_sq += ($4 * $4)
            b_sum += $5; b_sq += ($5 * $5)
            o_sum += $6; o_sq += ($6 * $6)
        }
        END {
            if (n == 0) {
                print "batch_size,phase,mean_s,std_s,num_samples"
                exit
            }
            step_mean = step_sum / n
            f_mean = f_sum / n
            b_mean = b_sum / n
            o_mean = o_sum / n

            step_std = sqrt((step_sq / n) - (step_mean * step_mean))
            f_std = sqrt((f_sq / n) - (f_mean * f_mean))
            b_std = sqrt((b_sq / n) - (b_mean * b_mean))
            o_std = sqrt((o_sq / n) - (o_mean * o_mean))

            if (step_std < 0) step_std = 0
            if (f_std < 0) f_std = 0
            if (b_std < 0) b_std = 0
            if (o_std < 0) o_std = 0

            print "batch_size,phase,mean_s,std_s,num_samples"
            printf "%s,step,%.6f,%.6f,%d\n", BATCH, step_mean, step_std, n
            printf "%s,forward,%.6f,%.6f,%d\n", BATCH, f_mean, f_std, n
            printf "%s,backward,%.6f,%.6f,%d\n", BATCH, b_mean, b_std, n
            printf "%s,optimizer,%.6f,%.6f,%d\n", BATCH, o_mean, o_std, n
        }
    ' "${files[@]}" > "${out_csv}"
}

aggregate_timeline_summary() {
    local batch_size="$1"
    local out_csv="$2"
    local files=()

    while IFS= read -r f; do
        files+=("$f")
    done < <(find "${RESULTS_DIR}" -type f -path "*/batch${batch_size}_run*/metrics/bert_run_timeline.csv" | sort)

    if [[ ${#files[@]} -eq 0 ]]; then
        echo "No bert_run_timeline.csv found for batch ${batch_size}" >&2
        exit 1
    fi

    local tmp_unsorted
    tmp_unsorted=$(mktemp)

    awk -F',' '
        FNR == 1 { next }
        {
            bin_idx = int(($1 / 0.5) + 0.5)
            t = bin_idx * 0.5

            n[t] += 1

            gpu_sum[t] += $2; gpu_sq[t] += ($2 * $2)
            gpumem_sum[t] += $4; gpumem_sq[t] += ($4 * $4)
            cpu_sum[t] += $5; cpu_sq[t] += ($5 * $5)
        }
        END {
            print "timeline_s,gpu_util_pct_mean,gpu_util_pct_std,gpu_mem_mb_mean,gpu_mem_mb_std,cpu_util_pct_mean,cpu_util_pct_std,num_samples"
            for (t in n) {
                gpu_mean = gpu_sum[t] / n[t]
                gpumem_mean = gpumem_sum[t] / n[t]
                cpu_mean = cpu_sum[t] / n[t]

                gpu_std = sqrt((gpu_sq[t] / n[t]) - (gpu_mean * gpu_mean))
                gpumem_std = sqrt((gpumem_sq[t] / n[t]) - (gpumem_mean * gpumem_mean))
                cpu_std = sqrt((cpu_sq[t] / n[t]) - (cpu_mean * cpu_mean))

                if (gpu_std < 0) gpu_std = 0
                if (gpumem_std < 0) gpumem_std = 0
                if (cpu_std < 0) cpu_std = 0

                printf "%.1f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d\n", t, gpu_mean, gpu_std, gpumem_mean, gpumem_std, cpu_mean, cpu_std, n[t]
            }
        }
    ' "${files[@]}" > "${tmp_unsorted}"

    {
        head -n 1 "${tmp_unsorted}"
        tail -n +2 "${tmp_unsorted}" | sort -t',' -k1,1n
    } > "${out_csv}"

    rm -f "${tmp_unsorted}"
}

for batch_size in "${BATCH_SIZES[@]}"; do
    repeat_val=$(get_repeat_for_batch "${batch_size}")
    run1_s=""
    run2_s=""
    run3_s=""

    for run in $(seq 1 "${RUNS_PER_BATCH}"); do
        if [[ "${first_run}" == "false" ]]; then
            echo "[E3] Cooling down for ${COOLDOWN_S}s before next run..."
            sleep "${COOLDOWN_S}"
        fi
        first_run=false

        run_id="batch${batch_size}_run${run}"
        run_dir="${RESULTS_DIR}/${run_id}"
        metrics_dir="${run_dir}/metrics"
        mkdir -p "${run_dir}" "${metrics_dir}"

        log_file="${LOG_DIR}/${run_id}.log"
        run_csv="${run_dir}/run_metadata.csv"

        echo "[E3] Starting ${run_id} (batch_size=${batch_size}, repeat=${repeat_val}, trainer_stats=simple)"

        start_ns=$(date +%s%N)
        {
            echo "[E3] Command start: $(date -Iseconds)"
            echo "[E3] Command: BERT_STATS_OUTPUT_DIR=${metrics_dir} ${SCRIPTS_DIR}/srun.sh --logging.level INFO --model bert --data bert --trainer simple --batch_size ${batch_size} --learning_rate 1e-4 --trainer_stats simple --data_configs.bert.repeat ${repeat_val} --data_configs.bert.n 0"
            run_e3_command "${batch_size}" "${repeat_val}" "${metrics_dir}"
            echo "[E3] Command end: $(date -Iseconds)"
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

        results_csv="${metrics_dir}/bert_run_results.csv"
        timeline_csv="${metrics_dir}/bert_run_timeline.csv"

        if [[ ! -f "${results_csv}" ]]; then
            echo "[E3] Missing ${results_csv}" >&2
            exit 1
        fi
        if [[ ! -f "${timeline_csv}" ]]; then
            echo "[E3] Missing ${timeline_csv}" >&2
            exit 1
        fi

        {
            echo "batch_size,repeat,run,trainer_stats,train_time_s,wall_clock_s,steps,target_seconds,results_csv,timeline_csv"
            echo "${batch_size},${repeat_val},${run},simple,${train_time_s},${wall_clock_s},${steps},${TARGET_SECONDS},${results_csv},${timeline_csv}"
        } > "${run_csv}"

        ts=$(date -Iseconds)
        echo "${ts},${batch_size},${repeat_val},${run},${wall_clock_s},${train_time_s},${steps},${results_csv},${timeline_csv},${run_dir},${log_file}" >> "${RUNS_CSV}"

        if [[ "${run}" == "1" ]]; then run1_s="${train_time_s}"; fi
        if [[ "${run}" == "2" ]]; then run2_s="${train_time_s}"; fi
        if [[ "${run}" == "3" ]]; then run3_s="${train_time_s}"; fi

        echo "[E3] Finished ${run_id}: train_time_s=${train_time_s}, steps=${steps}"
    done

    if [[ "${RUNS_PER_BATCH}" -ne 3 ]]; then
        avg_s=""
        std_s=""
    else
        avg_s=$(calc_mean "${run1_s}" "${run2_s}" "${run3_s}")
        std_s=$(calc_std "${run1_s}" "${run2_s}" "${run3_s}")
    fi

    echo "${batch_size},${repeat_val},${run1_s},${run2_s},${run3_s},${avg_s},${std_s},${TARGET_SECONDS}" >> "${TIME_SUMMARY_CSV}"

    batch_dir="${RESULTS_DIR}/batch${batch_size}"
    mkdir -p "${batch_dir}"
    aggregate_phase_summary "${batch_size}" "${batch_dir}/phase_summary.csv"
    aggregate_timeline_summary "${batch_size}" "${batch_dir}/timeline_summary.csv"

    echo "[E3] Batch ${batch_size} summary written: ${batch_dir}/phase_summary.csv, ${batch_dir}/timeline_summary.csv"
done

echo "[E3] Done. Artifacts:"
echo "  - ${RUNS_CSV}"
echo "  - ${TIME_SUMMARY_CSV}"
echo "  - ${RESULTS_DIR}/batch*/phase_summary.csv"
echo "  - ${RESULTS_DIR}/batch*/timeline_summary.csv"
echo "  - ${RESULTS_DIR}/batch*_run*/metrics/bert_run_results.csv"
echo "  - ${RESULTS_DIR}/batch*_run*/metrics/bert_run_timeline.csv"
echo "  - ${LOG_DIR}/batch*_run*.log"
