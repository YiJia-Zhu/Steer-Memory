#!/usr/bin/env bash
set -euo pipefail
set -m

# Cross-dataset generalization runner.
# - Uses memory from main experiments (default run_name prefix: "main").
# - For each (model, memory_dataset), runs eval on other datasets using that memory.
# - Applies chosen_params_for_main.csv with tier=high (the "high标准").
# - Keeps GPU scheduling logic consistent with run_main_experiments.sh (one job per GPU).

# =========================
# User-editable (TOP)
# =========================
# GPUs to use (one concurrent job per GPU).
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"

# Models: <model_key>|<name_or_path>|<tensor_parallel_size>|<max_num_seqs>
# Keep list form even if running a single model for easy edits.
MODEL_SPECS=(
  "ds_r1_qwen_7b|huggingface_models/DeepSeek-R1-Distill-Qwen-7B|1|256"
  # "ds_r1_qwen_1p5b|huggingface_models/DeepSeek-R1-Distill-Qwen-1.5B|1|256"
  # "qwen2p5_3b|huggingface_models/Qwen2.5-3B-Instruct|1|256"
  # "qwen2p5_7b|huggingface_models/Qwen2.5-7B-Instruct|1|128"
)

# Datasets: <dataset>|<prompt_template>|<train_split>|<eval_split>|<max_train>|<max_eval>|<T_max>|<max_model_len>
DATASET_SPECS=(
  "math500|math_0shot|test|test|100|400|16384|16384"
  "aime_2024|math_0shot|train|train|10|20|16384|16384"
  "amc23|math_0shot|test|test|10|30|16384|16384"
  "aime25|math_0shot|test|test|10|20|16384|16384"
  "arc-c|arc_0shot|train|validation|100|null|1024|4096" # 1.12k 299
  "openbookqa|arc_0shot|train|validation|100|null|1024|4096" # 4k 500 
  # "gsm8k|gsm8k_0shot|train|test|100|null|2048|4096"
  # "commonsense_qa|arc_0shot|train|validation|100|null|1024|4096" # 9k 1k
)

# Base config template.
BASE_CFG="${BASE_CFG:-configs/default}"

# Run naming.
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-generalization}"
# Memory source run prefix (main experiments).
SOURCE_RUN_PREFIX="${SOURCE_RUN_PREFIX:-main}"
# Shared run_id for this batch.
EXP_ID="${EXP_ID:-$(date +%Y%m%d_%H%M%S)}"

# Stage plan (default eval-only since memory is reused).
STAGES="${STAGES:-eval}"

# Eval methods toggle (kept aligned with main script).
RUN_GREEDY="${RUN_GREEDY:-1}"
RUN_ESM="${RUN_ESM:-1}"

# Use chosen params csv (tier high = “high标准”).
USE_CHOSEN_PARAMS="${USE_CHOSEN_PARAMS:-1}"
CHOSEN_PARAMS_CSV="${CHOSEN_PARAMS_CSV:-analysis/chosen_params_for_main.csv}"
CHOSEN_PARAMS_TIER="${CHOSEN_PARAMS_TIER:-high}"

# Optional overrides for target dataset budgets.
MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-}"
MAX_EVAL_EXAMPLES="${MAX_EVAL_EXAMPLES:-}"

# Where to dump generated configs / logs.
OUT_CFG_DIR="${OUT_CFG_DIR:-configs/_generalization_generated/${EXP_ID}}"
LOG_DIR="${LOG_DIR:-}"

# Resume / listing controls.
DRY_RUN="${DRY_RUN:-0}"
LIST_JOBS="${LIST_JOBS:-0}"
RESUME_FROM="${RESUME_FROM:-${RESUME:-0}}"   # 0-based index across all jobs

# Execution method.
USE_CONDA_RUN="${USE_CONDA_RUN:-0}"
CONDA_ENV="${CONDA_ENV:-easysteer}"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PY=(python)
if [[ "${USE_CONDA_RUN}" == "1" ]]; then
  PY=(conda run -n "${CONDA_ENV}" python)
fi

# Resolve BASE_CFG file.
if [[ ! -f "${BASE_CFG}" ]]; then
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

# Allow overrides via MODEL_SPECS_OVERRIDE / DATASET_SPECS_OVERRIDE (comma or newline separated).
if [[ -n "${MODEL_SPECS_OVERRIDE:-}" ]]; then
  MODEL_SPECS=()
  while IFS= read -r line; do
    line="$(echo "${line}" | xargs)"
    [[ -z "${line}" ]] && continue
    MODEL_SPECS+=( "${line}" )
  done <<< "$(printf "%s\n" "${MODEL_SPECS_OVERRIDE}" | tr ',' '\n')"
fi
if [[ -n "${DATASET_SPECS_OVERRIDE:-}" ]]; then
  DATASET_SPECS=()
  while IFS= read -r line; do
    line="$(echo "${line}" | xargs)"
    [[ -z "${line}" ]] && continue
    DATASET_SPECS+=( "${line}" )
  done <<< "$(printf "%s\n" "${DATASET_SPECS_OVERRIDE}" | tr ',' '\n')"
fi

# Resolve outputs.root_dir from base config.
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

declare -A PARAM_LAYER_MAP
declare -A PARAM_KSCALE_MAP
__param_rows=()
missing_param_warned=0

if [[ "${USE_CHOSEN_PARAMS}" != "0" ]]; then
  if [[ ! -f "${CHOSEN_PARAMS_CSV}" ]]; then
    echo "[error] CHOSEN_PARAMS_CSV not found: ${CHOSEN_PARAMS_CSV}" >&2
    exit 2
  fi

  mapfile -t __param_rows < <(CHOSEN_PARAMS_CSV="${CHOSEN_PARAMS_CSV}" CHOSEN_PARAMS_TIER="${CHOSEN_PARAMS_TIER}" "${PY[@]}" - <<'PY'
import os
from pathlib import Path
import pandas as pd

csv_path = Path(os.environ["CHOSEN_PARAMS_CSV"])
tier = str(os.environ.get("CHOSEN_PARAMS_TIER", "high")).strip()
df = pd.read_csv(csv_path)
df = df[df["tier"] == tier]
if df.empty:
    raise SystemExit(f"No rows found in {csv_path} for tier={tier}")

for _, row in df.iterrows():
    model = str(row["models"]).strip().replace("\\", "/").split("/")[-1]
    dataset = str(row["dataset"]).strip()
    offline = str(row["offline_candidate_layers"]).strip()
    kscale = str(row["online_k_scale"]).strip()
    print(f"{model}|{dataset}|{offline}|{kscale}")
PY
  )

  for line in "${__param_rows[@]}"; do
    IFS='|' read -r m d l k <<< "${line}"
    key="${m}|${d}"
    PARAM_LAYER_MAP["${key}"]="${l}"
    PARAM_KSCALE_MAP["${key}"]="${k}"
  done
fi

has_memory() {
  local run_dir="$1"
  [[ -d "${run_dir}/memory" ]] || return 1
  if compgen -G "${run_dir}/memory/*.npy" >/dev/null || compgen -G "${run_dir}/memory/*.jsonl" >/dev/null; then
    return 0
  fi
  # Allow empty but present memory dir.
  return 0
}

resolve_memory_run_dir() {
  local run_name="$1"
  local run_root="${OUTPUTS_ROOT}/${run_name}"
  if [[ ! -d "${run_root}" ]]; then
    echo "[error] memory run_root not found: ${run_root}" >&2
    return 1
  }

  local candidates=()
  if [[ -f "${run_root}/LATEST" ]]; then
    local latest
    latest="$(cat "${run_root}/LATEST" | xargs || true)"
    if [[ -n "${latest}" ]]; then
      candidates+=( "${latest}" )
    fi
  fi
  while IFS= read -r d; do
    d="$(basename "${d}")"
    candidates+=( "${d}" )
  done < <(find "${run_root}" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort -r)

  # Deduplicate while preserving order.
  local seen=()
  local ordered=()
  for c in "${candidates[@]}"; do
    [[ -z "${c}" ]] && continue
    local skip=0
    for s in "${seen[@]}"; do
      if [[ "${s}" == "${c}" ]]; then
        skip=1
        break
      fi
    done
    if [[ "${skip}" == "0" ]]; then
      ordered+=( "${c}" )
      seen+=( "${c}" )
    fi
  done

  for rid in "${ordered[@]}"; do
    local cand="${run_root}/${rid}"
    if [[ -d "${cand}" ]] && has_memory "${cand}"; then
      echo "${cand}"
      return 0
    fi
  done

  echo "[error] no memory found under ${run_root} (checked newest to oldest)" >&2
  return 1
}

write_cfg() {
  local out_cfg="$1"
  local run_name="$2"
  local model_path="$3"
  local model_tp="$4"
  local max_num_seqs="$5"
  local target_dataset="$6"
  local target_prompt="$7"
  local target_train_split="$8"
  local target_eval_split="$9"
  local target_max_train="${10}"
  local target_max_eval="${11}"
  local target_tmax="${12}"
  local target_max_model_len="${13}"
  local offline_layer="${14:-}"
  local online_k_scale="${15:-}"
  local artifact_run_dir="${16}"

  env BASE_CFG="${BASE_CFG}" OUT_CFG="${out_cfg}" RUN_NAME="${run_name}" \
    MODEL_PATH="${model_path}" MODEL_TP="${model_tp}" MAX_NUM_SEQS="${max_num_seqs}" \
    TARGET_DATASET="${target_dataset}" TARGET_PROMPT="${target_prompt}" \
    TARGET_TRAIN_SPLIT="${target_train_split}" TARGET_EVAL_SPLIT="${target_eval_split}" \
    TARGET_MAX_TRAIN="${target_max_train}" TARGET_MAX_EVAL="${target_max_eval}" \
    TARGET_TMAX="${target_tmax}" TARGET_MAX_MODEL_LEN="${target_max_model_len}" \
    OFFLINE_LAYER="${offline_layer}" ONLINE_K_SCALE="${online_k_scale}" \
    RUN_GREEDY="${RUN_GREEDY}" RUN_ESM="${RUN_ESM}" \
    GLOBAL_MAX_TRAIN="${MAX_TRAIN_EXAMPLES}" GLOBAL_MAX_EVAL="${MAX_EVAL_EXAMPLES}" \
    ARTIFACT_RUN_DIR="${artifact_run_dir}" \
    "${PY[@]}" - <<'PY'
import os
from pathlib import Path
import yaml

base_cfg = Path(os.environ["BASE_CFG"])
out_cfg = Path(os.environ["OUT_CFG"])

with base_cfg.open("r", encoding="utf-8") as f:
    raw = yaml.safe_load(f) or {}
if not isinstance(raw, dict):
    raise TypeError(f"Config root must be a dict, got {type(raw)}")

raw.setdefault("outputs", {})
raw["outputs"]["run_name"] = os.environ["RUN_NAME"]
raw["outputs"]["run_id"] = os.environ.get("EXP_ID", os.environ.get("RUN_ID"))

raw.setdefault("model", {})
raw["model"]["name_or_path"] = os.environ["MODEL_PATH"]
tp_s = str(os.environ.get("MODEL_TP", "__KEEP__")).strip()
if tp_s and tp_s != "__KEEP__":
    raw["model"]["tensor_parallel_size"] = int(tp_s)
max_num_seqs = str(os.environ.get("MAX_NUM_SEQS", "__KEEP__")).strip()
if max_num_seqs and max_num_seqs != "__KEEP__":
    raw["model"]["max_num_seqs"] = int(max_num_seqs)

raw.setdefault("task", {})
raw["task"]["dataset"] = os.environ["TARGET_DATASET"]
raw["task"]["train_split"] = os.environ["TARGET_TRAIN_SPLIT"]
raw["task"]["eval_split"] = os.environ["TARGET_EVAL_SPLIT"]

def _maybe_int(s: str):
    s = (s or "").strip()
    if s == "" or s == "__KEEP__":
        return "__KEEP__"
    if s.lower() in {"none", "null"}:
        return None
    return int(s)

g_train = _maybe_int(os.environ.get("GLOBAL_MAX_TRAIN", ""))
g_eval = _maybe_int(os.environ.get("GLOBAL_MAX_EVAL", ""))
max_train = _maybe_int(os.environ.get("TARGET_MAX_TRAIN", "__KEEP__"))
max_eval = _maybe_int(os.environ.get("TARGET_MAX_EVAL", "__KEEP__"))

if g_train != "__KEEP__":
    raw["task"]["max_train_examples"] = g_train
elif max_train != "__KEEP__":
    raw["task"]["max_train_examples"] = max_train

if g_eval != "__KEEP__":
    raw["task"]["max_eval_examples"] = g_eval
elif max_eval != "__KEEP__":
    raw["task"]["max_eval_examples"] = max_eval

raw.setdefault("prompt", {})
raw["prompt"]["template"] = os.environ["TARGET_PROMPT"]

raw.setdefault("decode", {})
tmax = _maybe_int(os.environ.get("TARGET_TMAX", "__KEEP__"))
if tmax != "__KEEP__":
    raw["decode"]["max_new_tokens"] = int(tmax) if tmax is not None else None
    raw.setdefault("offline_mine", {})
    raw["offline_mine"]["max_new_tokens"] = int(tmax) if tmax is not None else None

max_model_len = _maybe_int(os.environ.get("TARGET_MAX_MODEL_LEN", "__KEEP__"))
if max_model_len != "__KEEP__":
    raw.setdefault("model", {})
    raw["model"]["max_model_len"] = int(max_model_len) if max_model_len is not None else None

off_layer = str(os.environ.get("OFFLINE_LAYER", "")).strip()
if off_layer != "":
    raw.setdefault("offline_mine", {})
    raw["offline_mine"]["candidate_layers"] = [off_layer]

online_k = str(os.environ.get("ONLINE_K_SCALE", "")).strip()
if online_k != "":
    raw.setdefault("online", {})
    raw["online"]["k_scale"] = float(online_k)

methods = []
if str(os.environ.get("RUN_GREEDY", "1")).strip() not in {"0", "false", "False"}:
    methods.append("greedy")
if str(os.environ.get("RUN_ESM", "1")).strip() not in {"0", "false", "False"}:
    methods.append("esm")
if not methods:
    raise ValueError("Both RUN_GREEDY and RUN_ESM are disabled; nothing to run.")
raw.setdefault("eval", {})
raw["eval"]["methods"] = methods
raw["eval"]["ablations"] = []

artifact = os.environ.get("ARTIFACT_RUN_DIR", "").strip()
if not artifact:
    raise ValueError("ARTIFACT_RUN_DIR is required for cross generalization")
raw["eval"]["artifact_run_dir"] = artifact

out_cfg.parent.mkdir(parents=True, exist_ok=True)
with out_cfg.open("w", encoding="utf-8") as f:
    yaml.safe_dump(raw, f, allow_unicode=True, sort_keys=False)
PY
}

jobs=()  # "IDX|RUN_NAME|CFG_PATH|CMD"
job_idx=0

if [[ "${LIST_JOBS}" == "1" ]]; then
  echo "[list_jobs] 1"
  echo "[ordering] model -> memory_dataset -> target_dataset"
  echo
fi

for mspec in "${MODEL_SPECS[@]}"; do
  IFS='|' read -r model_key model_path model_tp model_max_num_seqs <<< "${mspec}"
  model_key_s="$(sanitize "${model_key}")"
  model_base="$(basename "${model_path}")"
  model_max_num_seqs="${model_max_num_seqs:-__KEEP__}"

  for mem_spec in "${DATASET_SPECS[@]}"; do
    IFS='|' read -r mem_dataset mem_prompt mem_train mem_eval mem_max_train mem_max_eval mem_tmax mem_max_model_len <<< "${mem_spec}"
    mem_dataset_s="$(sanitize "${mem_dataset}")"

    offline_layer=""
    online_k_scale=""
    if [[ "${USE_CHOSEN_PARAMS}" != "0" ]]; then
      param_key="${model_base}|${mem_dataset}"
      offline_layer="${PARAM_LAYER_MAP["${param_key}"]:-}"
      online_k_scale="${PARAM_KSCALE_MAP["${param_key}"]:-}"
      if [[ "${LIST_JOBS}" != "1" && ( -z "${offline_layer}" || -z "${online_k_scale}" ) && "${missing_param_warned}" == "0" ]]; then
        echo "[warn] missing chosen params for some jobs (fallback to base config values)" >&2
        missing_param_warned=1
      fi
    fi

    source_run_name="$(sanitize "${SOURCE_RUN_PREFIX}")_${model_key_s}_${mem_dataset_s}"
    if [[ "${LIST_JOBS}" != "1" ]]; then
      mem_dir="$(resolve_memory_run_dir "${source_run_name}")" || exit 2
    fi

    for tgt_spec in "${DATASET_SPECS[@]}"; do
      IFS='|' read -r tgt_dataset tgt_prompt tgt_train tgt_eval tgt_max_train tgt_max_eval tgt_tmax tgt_max_model_len <<< "${tgt_spec}"
      tgt_dataset_s="$(sanitize "${tgt_dataset}")"

      if [[ "${tgt_dataset}" == "${mem_dataset}" ]]; then
        # Only cross (memory on other datasets).
        continue
      fi

      run_name="$(sanitize "${RUN_NAME_PREFIX}")_${model_key_s}_mem_${mem_dataset_s}_on_${tgt_dataset_s}"
      cfg_path="${OUT_CFG_DIR}/${run_name}.yaml"

      if [[ "${LIST_JOBS}" == "1" ]]; then
        printf "%5d | model=%s | mem=%s | target=%s | run_name=%s | run_id=%s | stages=%s\n" \
          "${job_idx}" "${model_key_s}" "${mem_dataset_s}" "${tgt_dataset_s}" "${run_name}" "${EXP_ID}" "${STAGES}"
        job_idx=$((job_idx + 1))
        continue
      fi

      if (( job_idx < RESUME_FROM )); then
        job_idx=$((job_idx + 1))
        continue
      fi

      write_cfg \
        "${cfg_path}" "${run_name}" "${model_path}" "${model_tp}" "${model_max_num_seqs}" \
        "${tgt_dataset}" "${tgt_prompt}" "${tgt_train}" "${tgt_eval}" \
        "${tgt_max_train}" "${tgt_max_eval}" "${tgt_tmax}" "${tgt_max_model_len}" \
        "${offline_layer}" "${online_k_scale}" "${mem_dir}"

      cmd="cd \"${REPO_ROOT}\""
      for st in "${STAGE_LIST[@]}"; do
        st="$(echo "${st}" | xargs)"
        [[ -z "${st}" ]] && continue
        cmd+=" && ${PY[*]} run.py --config \"${cfg_path}\" --run-id \"${EXP_ID}\" ${st}"
      done

      jobs+=( "${job_idx}|${run_name}|${cfg_path}|${cmd}" )
      job_idx=$((job_idx + 1))
    done
  done
done

total_jobs="${job_idx}"

if [[ "${LIST_JOBS}" == "1" ]]; then
  echo
  echo "[n_jobs_total] ${total_jobs}"
  exit 0
fi

if (( ${#jobs[@]} == 0 )); then
  echo "[no jobs] RESUME_FROM=${RESUME_FROM} (n_jobs_total=${total_jobs})"
  exit 0
fi

echo "[mode] cross generalization"
echo "[base_cfg] ${BASE_CFG}"
echo "[run_name_prefix] ${RUN_NAME_PREFIX}"
echo "[source_run_prefix] ${SOURCE_RUN_PREFIX}"
echo "[exp_id] ${EXP_ID}"
echo "[stages] ${STAGES}"
echo "[methods] RUN_GREEDY=${RUN_GREEDY} RUN_ESM=${RUN_ESM}"
if [[ "${USE_CHOSEN_PARAMS}" != "0" ]]; then
  echo "[chosen_params] ${CHOSEN_PARAMS_CSV} (tier=${CHOSEN_PARAMS_TIER})"
else
  echo "[chosen_params] disabled (using base config values)"
fi
echo "[gpus] ${GPUS}"
echo "[n_jobs] ${#jobs[@]}"
echo "[outputs_root] ${OUTPUTS_ROOT}"
echo "[out_cfg_dir] ${OUT_CFG_DIR}"
if [[ -n "${LOG_DIR}" ]]; then
  echo "[log_dir] ${LOG_DIR}"
else
  echo "[log_dir] per-run (outputs/<run_name>/<run_id>/logs/stdout.log)"
fi
echo

if [[ "${DRY_RUN}" == "1" ]]; then
  for j in "${jobs[@]}"; do
    IFS='|' read -r idx rn cfg cmd <<< "${j}"
    echo "[dry_run] idx=${idx} run=${rn} cfg=${cfg} cmd=${cmd}"
  done
  exit 0
fi

available_gpus=( "${GPU_LIST[@]}" )
declare -A pid_to_gpu=()
declare -A pid_to_name=()
failed=0
failed_jobs=()

EVENT_TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/cross_generalization.${EXP_ID}.XXXXXX")"
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

    IFS='|' read -r idx name cfg cmd <<< "${job}"
    safe_name="$(printf "%05d__%s" "${idx}" "${name}")"
    start_job "${gpu}" "${safe_name}" "${cfg}" "${cmd}" "${name}" "${EXP_ID}"
  done

  if [[ "${#pid_to_gpu[@]}" -gt 0 ]]; then
    done_pid=""
    rc=0
    if IFS=' ' read -r done_pid rc <"${EVENT_FIFO}"; then
      :
    else
      echo "[error] failed to read job completion event" >&2
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
  echo "[all done] exp_id=${EXP_ID} (failed=${failed}/${#jobs[@]})" >&2
  for j in "${failed_jobs[@]}"; do
    echo "  - ${j}" >&2
  done
  exit 1
fi
echo "[all done] exp_id=${EXP_ID}"
