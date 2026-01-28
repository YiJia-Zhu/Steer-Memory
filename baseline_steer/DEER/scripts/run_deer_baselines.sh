#!/usr/bin/env bash
set -euo pipefail

# GPUs to use (comma-separated). Jobs rotate across this list.
GPUS="${GPUS:-0,1,2,3,4,5,6}"

# Root of Steer-Memory-114 (override via env STEER_ROOT if your path differs)
STEER_ROOT="${STEER_ROOT:-/private/zhenningshi/Steer-Memory-114}"

# Model specs: <model_key>|<model_path>|<script>|<dtype>|<policy>
# script is typically vllm-deer.py or vllm-deer-qwen3.py
MODEL_SPECS=(
  "ds_r1_qwen_1p5b|${STEER_ROOT}/huggingface_models/DeepSeek-R1-Distill-Qwen-1.5B|vllm-deer.py|bfloat16|avg1"
  # "qwen2p5_3b|${STEER_ROOT}/huggingface_models/Qwen2.5-3B-Instruct|vllm-deer.py|bfloat16|avg1"
  # "ds_r1_qwen_7b|${STEER_ROOT}/huggingface_models/DeepSeek-R1-Distill-Qwen-7B|vllm-deer.py|bfloat16|avg1"
  # "qwen2p5_7b|${STEER_ROOT}/huggingface_models/Qwen2.5-7B-Instruct|vllm-deer.py|bfloat16|avg1"
)

# Dataset specs from Steer-Memory (unused fields kept for clarity)
# <dataset>|<prompt_template>|<train_split>|<eval_split>|<max_train>|<max_eval>|<T_max>|<max_model_len>
DATASET_SPECS=(
  # "math500|math_0shot|test|test|100|400|16384|16384"
  # "aime_2024|math_0shot|train|train|10|20|16384|16384"
  # "amc23|math_0shot|test|test|10|30|16384|16384"
  # "aime25|math_0shot|test|test|10|20|16384|16384"
  "arc-c|arc_0shot|train|validation|100|null|1024|4096"
  # "openbookqa|arc_0shot|train|validation|100|null|1024|4096"
)

# Roots (override via env if needed)
DATASET_ROOT="${DATASET_ROOT:-${STEER_ROOT}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/private/zhenningshi/baseline_steer/DEER/outputs}"
DRY_RUN="${DRY_RUN:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-}"

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
if [[ "${#GPU_LIST[@]}" -eq 0 ]]; then
  echo "[error] No GPUs configured via GPUS env." >&2
  exit 1
fi

sanitize() {
  local s="$1"
  s="$(echo "${s}" | tr '[:upper:]' '[:lower:]')"
  s="${s//[^a-z0-9]/_}"
  s="${s//__/_}"
  s="${s#_}"
  s="${s%_}"
  echo "${s}"
}

jobs=()  # "idx|run_name|cmd_str"
job_idx=0
for model_spec in "${MODEL_SPECS[@]}"; do
  IFS='|' read -r MODEL_KEY MODEL_PATH MODEL_SCRIPT MODEL_DTYPE MODEL_POLICY <<< "${model_spec}"
  MODEL_SCRIPT="${MODEL_SCRIPT:-vllm-deer.py}"
  for ds_spec in "${DATASET_SPECS[@]}"; do
    IFS='|' read -r DATASET PROMPT_TEMPLATE TRAIN_SPLIT EVAL_SPLIT MAX_TRAIN MAX_EVAL TMAX MAX_MODEL_LEN <<< "${ds_spec}"

    cmd=(/opt/conda/envs/easysteer/bin/python "${SCRIPT_ROOT}/${MODEL_SCRIPT}"
      --model_name_or_path "${MODEL_PATH}"
      --dataset_dir "${DATASET_ROOT}"
      --dataset "${DATASET}"
      --split "${EVAL_SPLIT}"
      --output_path "${OUTPUT_ROOT}"
    )

    if [[ -n "${TMAX}" && "${TMAX}" != "null" && "${TMAX}" != "__KEEP__" ]]; then
      cmd+=(--max_generated_tokens "${TMAX}")
    fi
    if [[ -n "${MAX_EVAL}" && "${MAX_EVAL}" != "null" && "${MAX_EVAL}" != "__KEEP__" ]]; then
      cmd+=(--max_eval "${MAX_EVAL}")
    fi
    if [[ -n "${MAX_MODEL_LEN}" && "${MAX_MODEL_LEN}" != "null" && "${MAX_MODEL_LEN}" != "__KEEP__" ]]; then
      cmd+=(--model-context-len "${MAX_MODEL_LEN}")
    fi
    if [[ -n "${MODEL_DTYPE}" ]]; then
      cmd+=(--dtype "${MODEL_DTYPE}")
    fi
    if [[ -n "${MODEL_POLICY}" ]]; then
      cmd+=(--policy "${MODEL_POLICY}")
    fi

    run_name="$(sanitize "${MODEL_KEY}_${DATASET}_${EVAL_SPLIT}")"
    cmd_str="$(printf '%q ' "${cmd[@]}")"
    jobs+=( "${job_idx}|${run_name}|${cmd_str}" )
    echo "[$job_idx] MODEL=${MODEL_KEY} DATASET=${DATASET} SPLIT=${EVAL_SPLIT} SCRIPT=${MODEL_SCRIPT}"
    echo "      CMD: CUDA_VISIBLE_DEVICES=<gpu> ${cmd[*]}"

    job_idx=$((job_idx + 1))
  done
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[dry_run] planned ${#jobs[@]} job(s); exiting before launch."
  exit 0
fi

if (( ${#jobs[@]} == 0 )); then
  echo "[info] No jobs to run."
  exit 0
fi

echo "[launch] ${#jobs[@]} job(s) across ${#GPU_LIST[@]} GPU(s): ${GPUS}"
echo "[run_id] ${RUN_ID}"
if [[ -n "${LOG_DIR}" ]]; then
  echo "[log_dir] ${LOG_DIR}"
else
  echo "[log_dir] per-run under ${OUTPUT_ROOT}/<run_name>/${RUN_ID}/logs/stdout.log"
fi
echo

available_gpus=( "${GPU_LIST[@]}" )
declare -A pid_to_gpu=()
declare -A pid_to_name=()
failed=0
failed_jobs=()

EVENT_TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/deer_baselines.${RUN_ID}.XXXXXX")"
EVENT_FIFO="${EVENT_TMP_DIR}/events.fifo"
mkfifo "${EVENT_FIFO}"
cleanup_events() {
  rm -rf "${EVENT_TMP_DIR}"
}

terminate_jobs() {
  local pids=("${!pid_to_gpu[@]}")
  if (( ${#pids[@]} == 0 )); then
    return
  fi
  echo "[signal] terminating ${#pids[@]} running job(s)..." >&2
  for pid in "${pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  sleep 1
  for pid in "${pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
    fi
  done
}

on_signal() {
  local sig="$1"
  terminate_jobs
  cleanup_events
  if [[ "${sig}" == "INT" ]]; then
    exit 130
  fi
  if [[ "${sig}" == "TERM" ]]; then
    exit 143
  fi
  exit 1
}

trap cleanup_events EXIT
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM

start_job() {
  local gpu="$1"
  local name="$2"
  local cmd="$3"
  local log_path=""

  if [[ -n "${LOG_DIR}" ]]; then
    log_path="${LOG_DIR}/${name}.log"
    mkdir -p "${LOG_DIR}"
  else
    local run_dir="${OUTPUT_ROOT}/${name}/${RUN_ID}"
    local run_logs="${run_dir}/logs"
    mkdir -p "${run_logs}"
    log_path="${run_logs}/stdout.log"
  fi

  echo "[start][gpu=${gpu}] ${name}"
  echo "  log: ${log_path}"

  (
    set +e
    export CUDA_VISIBLE_DEVICES="${gpu}"
    bash -lc "${cmd}"
    rc=$?
    printf '%s %s\n' "${BASHPID}" "${rc}" >"${EVENT_FIFO}"
    exit "${rc}"
  ) >"${log_path}" 2>&1 &

  local pid="$!"
  pid_to_gpu["${pid}"]="${gpu}"
  pid_to_name["${pid}"]="${name}"
}

queue=( "${jobs[@]}" )
while [[ "${#queue[@]}" -gt 0 || "${#pid_to_gpu[@]}" -gt 0 ]]; do
  while [[ "${#queue[@]}" -gt 0 && "${#available_gpus[@]}" -gt 0 ]]; do
    job="${queue[0]}"
    queue=( "${queue[@]:1}" )

    gpu="${available_gpus[0]}"
    available_gpus=( "${available_gpus[@]:1}" )

    IFS='|' read -r idx run_name cmd <<< "${job}"
    safe_name="$(printf "%05d__%s" "${idx}" "${run_name}")"
    start_job "${gpu}" "${safe_name}" "${cmd}"
  done

  if [[ "${#pid_to_gpu[@]}" -gt 0 ]]; then
    done_pid=""
    rc=0
    if IFS=' ' read -r done_pid rc <"${EVENT_FIFO}"; then
      :
    else
      echo "[error] failed to read job completion event" >&2
      terminate_jobs
      exit 2
    fi
    wait "${done_pid}" >/dev/null 2>&1 || true

    if [[ -n "${done_pid}" ]]; then
      gpu="${pid_to_gpu[${done_pid}]}"
      name="${pid_to_name[${done_pid}]}"
      unset pid_to_gpu["${done_pid}"]
      unset pid_to_name["${done_pid}"]
      available_gpus+=( "${gpu}" )
      if (( rc != 0 )); then
        failed=$((failed + 1))
        failed_jobs+=( "${name}|gpu=${gpu}|rc=${rc}" )
        echo "[fail][gpu=${gpu}][rc=${rc}] ${name}" >&2
      else
        echo "[done][gpu=${gpu}] ${name}"
      fi
    fi
  fi
done

echo
if (( failed > 0 )); then
  echo "[all done] run_id=${RUN_ID} (failed=${failed}/${#jobs[@]})" >&2
  for j in "${failed_jobs[@]}"; do
    echo "  - ${j}" >&2
  done
  exit 1
fi
echo "[all done] run_id=${RUN_ID}"
