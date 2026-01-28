#!/usr/bin/env bash
set -euo pipefail
set -m

# =========================
# User-editable (TOP)
# =========================
# GPUs to use for parallel runs (one job per GPU). Edit as needed.
GPUS="${GPUS:-0,1,2,3,4,5,6}"
# GPUS="${GPUS:-0}"

# Baselines to run (comma-separated): self_consist, self_discover
BASELINES="${BASELINES:-self_consist,self_discover}"

# -------------------------
# Models to run
# Format: <model_key>|<name_or_path>|<tensor_parallel_size>|<max_num_seqs>
# -------------------------
MODEL_SPECS=(
  "ds_r1_qwen_1p5b|huggingface_models/DeepSeek-R1-Distill-Qwen-1.5B|1|512"
  "qwen2p5_3b|huggingface_models/Qwen2.5-3B-Instruct|1|512"
  "ds_r1_qwen_7b|huggingface_models/DeepSeek-R1-Distill-Qwen-7B|1|256"
  "qwen2p5_7b|huggingface_models/Qwen2.5-7B-Instruct|1|256"
)

# -------------------------
# Datasets to run (must be supported by esm/data/loaders.py and available locally)
# Format:
#   <dataset>|<prompt_template>|<train_split>|<eval_split>|<max_train>|<max_eval>|<T_max/max_new_token>|<max_model_len>
#
# Notes:
# - T_max is a shorthand for generation budget: decode.max_new_tokens (and offline_mine.max_new_tokens).
# - max_model_len is the vLLM context window cap: prompt tokens + generated tokens (affects KV cache/memory).
# - Use "__KEEP__" to keep the base-config value for that field.
# -------------------------
DATASET_SPECS=(
  "math500|math_0shot|test|test|100|400|16384|16384"
  "aime_2024|math_0shot|train|train|10|20|16384|16384"
  "amc23|math_0shot|test|test|10|30|16384|16384"
  "aime25|math_0shot|test|test|10|20|16384|16384"
  "arc-c|arc_0shot|train|validation|100|null|1024|4096" # 1.12k 299
  "openbookqa|arc_0shot|train|validation|100|null|1024|4096" # 4k 500
)

# -------------------------
# Runtime controls
# -------------------------

# Stages to run (comma-separated). Default: eval-only for baselines.
STAGES="${STAGES:-eval}"

# Base config template. Per-job configs are generated from this file.
BASE_CFG="${BASE_CFG:-configs/default}"

# Outputs: each job uses run_name = <RUN_NAME_PREFIX>_<baseline>_<model_key>_<dataset_key>
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-baseline}"

# Shared run_id prefix for this batch (helps grouping).
BASELINE_ID="${BASELINE_ID:-$(date +%Y%m%d_%H%M%S)}"

# If DRY_RUN=1, only generate configs and print the plan.
DRY_RUN="${DRY_RUN:-0}"

# Resume / listing controls.
# - RESUME_FROM: 0-based job index to start from (skips jobs < RESUME_FROM).
# - LIST_JOBS=1: print idx -> (baseline, model, dataset, run_name, run_id) and exit.
#
# Ordering is the nested-loop order in this script:
#   baseline -> model -> dataset
# Index formula (0-based):
#   idx = (b*n_models + m)*n_datasets + d
# where (b,m,d) are 0-based indices into BASELINES, MODEL_SPECS, DATASET_SPECS.
RESUME_FROM="${RESUME_FROM:-${RESUME:-0}}"
LIST_JOBS="${LIST_JOBS:-0}"

# Use current python by default (assumes you're already in the easysteer env).
# If you prefer forcing conda-run, set USE_CONDA_RUN=1.
USE_CONDA_RUN="${USE_CONDA_RUN:-0}"
CONDA_ENV="${CONDA_ENV:-easysteer}"

# Global optional overrides (leave empty to keep per-dataset / base config values).
MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-}"
MAX_EVAL_EXAMPLES="${MAX_EVAL_EXAMPLES:-}"

# =========================

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PY=(/opt/conda/envs/easysteer/bin/python)
if [[ "${USE_CONDA_RUN}" == "1" ]]; then
  PY=(conda run -n "${CONDA_ENV}" python)
fi

if [[ ! -f "${BASE_CFG}" ]]; then
  # Allow BASE_CFG=configs/default (no extension).
  if [[ -f "${BASE_CFG}.yaml" ]]; then
    BASE_CFG="${BASE_CFG}.yaml"
  elif [[ -f "${BASE_CFG}.yml" ]]; then
    BASE_CFG="${BASE_CFG}.yml"
  fi
fi

if [[ ! -f "${BASE_CFG}" ]]; then
  echo "[error] BASE_CFG not found: ${BASE_CFG}" >&2
  exit 2
fi

IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
if [[ "${#GPU_LIST[@]}" -le 0 ]]; then
  echo "[error] Empty GPUS list: ${GPUS}" >&2
  exit 2
fi

IFS=',' read -r -a STAGE_LIST <<< "${STAGES}"
if [[ "${#STAGE_LIST[@]}" -le 0 ]]; then
  echo "[error] Empty STAGES: ${STAGES}" >&2
  exit 2
fi

IFS=',' read -r -a BASELINE_LIST <<< "${BASELINES}"
if [[ "${#BASELINE_LIST[@]}" -le 0 ]]; then
  echo "[error] Empty BASELINES: ${BASELINES}" >&2
  exit 2
fi

n_models="${#MODEL_SPECS[@]}"
n_datasets="${#DATASET_SPECS[@]}"
n_baselines="${#BASELINE_LIST[@]}"
total_jobs=$(( n_baselines * n_models * n_datasets ))
if [[ ! "${RESUME_FROM}" =~ ^[0-9]+$ ]]; then
  echo "[error] RESUME_FROM must be a non-negative integer, got: ${RESUME_FROM}" >&2
  exit 2
fi

OUT_CFG_DIR="${OUT_CFG_DIR:-configs/_self_baselines/${BASELINE_ID}}"
LOG_DIR="${LOG_DIR:-}"

OUTPUTS_ROOT="$(
  BASE_CFG="${BASE_CFG}" "${PY[@]}" - <<'PY'
import os
from pathlib import Path

import yaml

base_cfg = Path(os.environ["BASE_CFG"])
raw = yaml.safe_load(base_cfg.read_text(encoding="utf-8")) or {}
root = (raw.get("outputs") or {}).get("root_dir") or "outputs"
print(Path(root).expanduser())
PY
)"
if [[ -z "${OUTPUTS_ROOT}" ]]; then
  OUTPUTS_ROOT="outputs"
fi
if [[ "${OUTPUTS_ROOT}" != /* ]]; then
  OUTPUTS_ROOT="${REPO_ROOT}/${OUTPUTS_ROOT}"
fi

mkdir -p "${OUT_CFG_DIR}"
if [[ -n "${LOG_DIR}" ]]; then
  mkdir -p "${LOG_DIR}"
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

write_cfg() {
  local out_cfg="$1"
  local run_name="$2"
  local baseline="$3"
  local model_path="$4"
  local tp="$5"
  local max_num_seqs="$6"
  local dataset="$7"
  local prompt_template="$8"
  local train_split="$9"
  local eval_split="${10}"
  local max_train="${11}"
  local max_eval="${12}"
  local tmax="${13}"
  local max_model_len="${14}"

  env BASE_CFG="${BASE_CFG}" OUT_CFG="${out_cfg}" RUN_NAME="${run_name}" BASELINE="${baseline}" \
    MODEL_PATH="${model_path}" MODEL_TP="${tp}" \
    MAX_NUM_SEQS="${max_num_seqs}" \
    DATASET="${dataset}" PROMPT_TEMPLATE="${prompt_template}" TRAIN_SPLIT="${train_split}" EVAL_SPLIT="${eval_split}" \
    MAX_TRAIN="${max_train}" MAX_EVAL="${max_eval}" TMAX="${tmax}" MAX_MODEL_LEN="${max_model_len}" \
    GLOBAL_MAX_TRAIN="${MAX_TRAIN_EXAMPLES}" GLOBAL_MAX_EVAL="${MAX_EVAL_EXAMPLES}" \
    "${PY[@]}" - <<'PY'
import os
from pathlib import Path

import yaml

base_cfg = Path(os.environ["BASE_CFG"])
out_cfg = Path(os.environ["OUT_CFG"])
baseline = os.environ["BASELINE"].strip().lower().replace("-", "_")

with base_cfg.open("r", encoding="utf-8") as f:
    raw = yaml.safe_load(f) or {}
if not isinstance(raw, dict):
    raise TypeError(f"Config root must be a dict, got {type(raw)}")

raw.setdefault("outputs", {})
raw["outputs"]["run_name"] = os.environ["RUN_NAME"]

raw.setdefault("model", {})
raw["model"]["name_or_path"] = os.environ["MODEL_PATH"]
tp_s = str(os.environ.get("MODEL_TP", "__KEEP__")).strip()
if tp_s and tp_s != "__KEEP__":
    raw["model"]["tensor_parallel_size"] = int(tp_s)

raw.setdefault("task", {})
raw["task"]["dataset"] = os.environ["DATASET"]
raw["task"]["train_split"] = os.environ["TRAIN_SPLIT"]
raw["task"]["eval_split"] = os.environ["EVAL_SPLIT"]

def _maybe_int(s: str):
    s = (s or "").strip()
    if s == "" or s == "__KEEP__":
        return "__KEEP__"
    if s.lower() in {"none", "null"}:
        return None
    return int(s)

# Global override (env) > per-dataset > base config
g_train = _maybe_int(os.environ.get("GLOBAL_MAX_TRAIN", ""))
g_eval = _maybe_int(os.environ.get("GLOBAL_MAX_EVAL", ""))
ds_train = _maybe_int(os.environ.get("MAX_TRAIN", "__KEEP__"))
ds_eval = _maybe_int(os.environ.get("MAX_EVAL", "__KEEP__"))

if g_train != "__KEEP__":
    raw["task"]["max_train_examples"] = g_train
elif ds_train != "__KEEP__":
    raw["task"]["max_train_examples"] = ds_train

if g_eval != "__KEEP__":
    raw["task"]["max_eval_examples"] = g_eval
elif ds_eval != "__KEEP__":
    raw["task"]["max_eval_examples"] = ds_eval

raw.setdefault("prompt", {})
tmpl = os.environ["PROMPT_TEMPLATE"]
if baseline == "self_discover" and not tmpl.lower().endswith("_self_discover"):
    tmpl = f"{tmpl}_self_discover"
raw["prompt"]["template"] = tmpl

raw.setdefault("decode", {})
tmax = _maybe_int(os.environ.get("TMAX", "__KEEP__"))
if tmax != "__KEEP__":
    raw["decode"]["max_new_tokens"] = int(tmax) if tmax is not None else None
    raw.setdefault("offline_mine", {})
    raw["offline_mine"]["max_new_tokens"] = int(tmax) if tmax is not None else None

max_model_len = _maybe_int(os.environ.get("MAX_MODEL_LEN", "__KEEP__"))
if max_model_len != "__KEEP__":
    raw.setdefault("model", {})
    raw["model"]["max_model_len"] = int(max_model_len) if max_model_len is not None else None

max_num_seqs = _maybe_int(os.environ.get("MAX_NUM_SEQS", "__KEEP__"))
if max_num_seqs != "__KEEP__":
    if max_num_seqs is None:
        raise ValueError("MAX_NUM_SEQS cannot be null/none")
    raw.setdefault("model", {})
    raw["model"]["max_num_seqs"] = int(max_num_seqs)

raw.setdefault("eval", {})
raw["eval"]["methods"] = [baseline]
raw["eval"]["ablations"] = []

out_cfg.parent.mkdir(parents=True, exist_ok=True)
with out_cfg.open("w", encoding="utf-8") as f:
    yaml.safe_dump(raw, f, allow_unicode=True, sort_keys=False)
PY
}

jobs=()  # "IDX|BASELINE|RUN_NAME|RID|CFG_PATH|CMD"
job_idx=0
if [[ "${LIST_JOBS}" == "1" ]]; then
  echo "[list_jobs] 1"
  echo "[ordering] baseline -> model -> dataset"
  echo "[note] set BASELINE_ID to match previous run_ids if needed"
  echo
fi
for bspec in "${BASELINE_LIST[@]}"; do
  baseline="$(sanitize "${bspec}")"
  rid="${BASELINE_ID}__${baseline}"

  for mspec in "${MODEL_SPECS[@]}"; do
    IFS='|' read -r model_key model_path model_tp model_max_num_seqs <<< "${mspec}"
    model_key="$(sanitize "${model_key}")"
    model_max_num_seqs="${model_max_num_seqs:-__KEEP__}"

    for dspec in "${DATASET_SPECS[@]}"; do
      IFS='|' read -r dataset prompt_template train_split eval_split max_train max_eval tmax max_model_len <<< "${dspec}"
      dataset_key="$(sanitize "${dataset}")"
      run_name="$(sanitize "${RUN_NAME_PREFIX}")_${baseline}_${model_key}_${dataset_key}"
      cfg_path="${OUT_CFG_DIR}/${run_name}.yaml"

      if [[ "${LIST_JOBS}" == "1" ]]; then
        printf "%5d | baseline=%s | model=%s | dataset=%s | run_name=%s | run_id=%s\n" \
          "${job_idx}" "${baseline}" "${model_key}" "${dataset_key}" "${run_name}" "${rid}"
        job_idx=$((job_idx + 1))
        continue
      fi

      if (( job_idx < RESUME_FROM )); then
        job_idx=$((job_idx + 1))
        continue
      fi

      write_cfg \
        "${cfg_path}" "${run_name}" "${baseline}" "${model_path}" "${model_tp}" "${model_max_num_seqs}" \
        "${dataset}" "${prompt_template}" "${train_split}" "${eval_split}" \
        "${max_train}" "${max_eval}" "${tmax}" "${max_model_len}"

      cmd="cd \"${REPO_ROOT}\""
      for st in "${STAGE_LIST[@]}"; do
        st="$(echo "${st}" | xargs)"
        if [[ -z "${st}" ]]; then
          continue
        fi
        cmd+=" && ${PY[*]} run.py --config \"${cfg_path}\" --run-id \"${rid}\" ${st}"
      done

      jobs+=( "${job_idx}|${baseline}|${run_name}|${rid}|${cfg_path}|${cmd}" )
      job_idx=$((job_idx + 1))
    done
  done
done

if [[ "${LIST_JOBS}" == "1" ]]; then
  echo
  echo "[n_jobs_total] ${total_jobs}"
  exit 0
fi

echo "[mode] self baselines (greedy-style)"
echo "[base_cfg] ${BASE_CFG}"
echo "[run_name_prefix] ${RUN_NAME_PREFIX}"
echo "[baseline_id] ${BASELINE_ID}"
echo "[stages] ${STAGES}"
echo "[baselines] ${BASELINES}"
echo "[resume_from] ${RESUME_FROM} (0-based index)"
echo "[gpus] ${GPUS}"
echo "[n_models] ${n_models}"
echo "[n_datasets] ${n_datasets}"
echo "[n_baselines] ${n_baselines}"
echo "[n_jobs_total] ${total_jobs}"
echo "[n_jobs] ${#jobs[@]}"
echo "[out_cfg_dir] ${OUT_CFG_DIR}"
echo "[outputs_root] ${OUTPUTS_ROOT}"
if [[ -n "${LOG_DIR}" ]]; then
  echo "[log_dir] ${LOG_DIR}"
else
  echo "[log_dir] per-run (outputs/<run_name>/<run_id>/logs/stdout.log)"
fi
echo

if (( ${#jobs[@]} == 0 )); then
  echo "[no jobs] RESUME_FROM=${RESUME_FROM} (n_jobs_total=${total_jobs})"
  exit 0
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[dry_run] 1 (configs generated; jobs not started)"
  exit 0
fi

available_gpus=( "${GPU_LIST[@]}" )
declare -A pid_to_gpu=()
declare -A pid_to_name=()
failed=0
failed_jobs=()

# Completion event queue (bash 4.x compatible replacement for `wait -n -p`).
EVENT_TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/self_baselines.${BASELINE_ID}.XXXXXX")"
EVENT_FIFO="${EVENT_TMP_DIR}/events.fifo"
mkfifo "${EVENT_FIFO}"
cleanup_events() {
  rm -rf "${EVENT_TMP_DIR}"
}

terminate_requested=0
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
  if [[ "${terminate_requested}" == "1" ]]; then
    exit 1
  fi
  terminate_requested=1
  echo "[signal] ${sig} received; terminating running jobs..." >&2
  terminate_jobs
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
  local cfg="$3"
  local cmd="$4"
  local run_name="$5"
  local run_id="$6"
  local log_path=""

  if [[ -n "${LOG_DIR}" ]]; then
    log_path="${LOG_DIR}/${name}.log"
    mkdir -p "${LOG_DIR}"
  else
    local run_dir="${OUTPUTS_ROOT}/${run_name}/${run_id}"
    local run_logs="${run_dir}/logs"
    mkdir -p "${run_logs}"
    log_path="${run_logs}/stdout.log"
  fi

  echo "[start][gpu=${gpu}] ${name}"
  echo "  cfg: ${cfg}"
  echo "  log: ${log_path}"

  (
    set +e
    export CUDA_VISIBLE_DEVICES="${gpu}"
    bash -lc "${cmd}"
    rc=$?
    # Use BASHPID (not $$) so the parent can match `$!`.
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

    IFS='|' read -r idx baseline run_name rid cfg cmd <<< "${job}"
    safe_name="$(printf "%05d__%s__%s" "${idx}" "${baseline}" "${run_name}")"
    start_job "${gpu}" "${safe_name}" "${cfg}" "${cmd}" "${run_name}" "${rid}"
  done

  if [[ "${#pid_to_gpu[@]}" -gt 0 ]]; then
    done_pid=""
    rc=0
    # Block until any job finishes and reports "<pid> <rc>".
    if IFS=' ' read -r done_pid rc <"${EVENT_FIFO}"; then
      :
    else
      echo "[error] failed to read job completion event" >&2
      exit 2
    fi
    # Reap to avoid zombies (ignore wait rc; we already captured rc).
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
  echo "[all done] baseline_id=${BASELINE_ID} (failed=${failed}/${#jobs[@]})" >&2
  for j in "${failed_jobs[@]}"; do
    echo "  - ${j}" >&2
  done
  exit 1
fi
echo "[all done] baseline_id=${BASELINE_ID}"

# -------------------------
# Summarize results into one CSV
# -------------------------
if [[ "${DRY_RUN}" != "1" ]]; then
  echo "[summary] writing consolidated CSV for baseline_id=${BASELINE_ID} ..."
  SUMMARY_OUT="${SUMMARY_OUT:-${OUTPUTS_ROOT}/_self_baselines/${BASELINE_ID}/summary.csv}"
  mkdir -p "$(dirname "${SUMMARY_OUT}")"
  python "${REPO_ROOT}/scripts/summarize_self_baselines.py" \
    --out-cfg-dir "${OUT_CFG_DIR}" \
    --outputs-root "${OUTPUTS_ROOT}" \
    --baseline-id "${BASELINE_ID}" \
    --summary-out "${SUMMARY_OUT}"
fi
