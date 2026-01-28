#!/usr/bin/env bash
set -euo pipefail
set -m

# Multi-GPU ablation runner (aligned with scripts/run_main_experiments.sh).
# - Maintains model/dataset lists.
# - Applies analysis/chosen_params_for_main.csv (tier-selectable).
# - Generates per-(model,dataset,ablation) configs under configs/_ablation_generated/<EXP_ID>/.
# - Writes outputs under outputs/<run_name>/<run_id>.
# - Default stage plan per ablation; override with STAGES env.
#
# Ablation keywords (built-in overrides):
#   random_mining     : offline_mine.method=random      (default stages: mine,select,memory,eval)
#   top1_memory       : offline_select.B=1, online.k_retrieve=1, online.L=1 (default full pipeline)
#   use_random_memory : online.variant=use_random_memory (default eval-only)
#   random_use_memory : online.variant=random_use_memory (default eval-only)
#   no_probing        : online.variant=no_probing        (default eval-only)
#   first_step_only   : online.max_tool_m=1              (default eval-only)
#
# Examples:
#   GPUS=0,1 EXP_ID=20250301_120000 bash scripts/run_ablation_experiments.sh
#   GPUS=0 STAGES=eval ABLATIONS="use_random_memory,no_probing" BASE_RUN_ID=latest bash scripts/run_ablation_experiments.sh

# =========================
# User-editable (TOP)
# =========================
GPUS="${GPUS:-0,1,2,3,4,5,6}"

# Models: <model_key>|<name_or_path>|<tensor_parallel_size>|<max_num_seqs>
MODEL_SPECS=(
  # Keep model_key aligned with main experiments so eval-only ablations can reuse artifacts.
  "ds_r1_qwen_7b|huggingface_models/DeepSeek-R1-Distill-Qwen-7B|1|128"
  # "qwen2p5_3b|huggingface_models/Qwen2.5-3B-Instruct|1|256"
)

# Datasets: <dataset>|<prompt_template>|<train_split>|<eval_split>|<max_train>|<max_eval>|<T_max>|<max_model_len>
DATASET_SPECS=(
  "math500|math_0shot|test|test|100|400|16384|16384"
  # "aime_2024|math_0shot|train|train|10|20|16384|16384"
  # "amc23|math_0shot|test|test|10|30|16384|16384"
  # "aime25|math_0shot|test|test|10|20|16384|16384"
  # "arc-c|arc_0shot|train|validation|100|null|1024|4096" # 1.12k 299
  # "openbookqa|arc_0shot|train|validation|100|null|1024|4096" # 4k 500 
  # "gsm8k|gsm8k_0shot|train|test|100|null|2048|4096"
  # "commonsense_qa|arc_0shot|train|validation|100|null|1024|4096" # 9k 1k
)

# Ablations to run (comma-separated from the list above).
# ABLATIONS="${ABLATIONS:-random_mining,top1_memory,use_random_memory,random_use_memory,no_probing,first_step_only}"
ABLATIONS="${ABLATIONS:-top1_memory}"

# Base config template.
BASE_CFG="${BASE_CFG:-configs/default}"

# Run naming.
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-ablation}"
BASE_RUN_NAME_PREFIX="${BASE_RUN_NAME_PREFIX:-main}"  # where to pull artifacts when skipping offline stages
EXP_ID="${EXP_ID:-$(date +%Y%m%d_%H%M%S)}"   # shared run_id

# Stage override (applies to all ablations if set).
# Defaults (per ablation) are described above.
STAGES="${STAGES:-}"

# Eval methods toggle.
RUN_GREEDY="${RUN_GREEDY:-0}"
RUN_ESM="${RUN_ESM:-1}"

# Use chosen params csv (tier=high by default).
USE_CHOSEN_PARAMS="${USE_CHOSEN_PARAMS:-1}"
CHOSEN_PARAMS_CSV="${CHOSEN_PARAMS_CSV:-analysis/chosen_params_for_main.csv}"
CHOSEN_PARAMS_TIER="${CHOSEN_PARAMS_TIER:-high}"

# Artifact source for eval-only ablations (no offline stages in stage plan).
BASE_RUN_ID="${BASE_RUN_ID:-latest}"

# Where to dump generated configs / logs.
OUT_CFG_DIR="${OUT_CFG_DIR:-configs/_ablation_generated/${EXP_ID}}"
LOG_DIR="${LOG_DIR:-}"

# Resume / listing.
DRY_RUN="${DRY_RUN:-0}"
LIST_JOBS="${LIST_JOBS:-0}"
RESUME_FROM="${RESUME_FROM:-0}"   # 0-based index across all (model,dataset,ablation)

# Execution method.
USE_CONDA_RUN="${USE_CONDA_RUN:-0}"
CONDA_ENV="${CONDA_ENV:-easysteer}"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PY=(/opt/conda/envs/easysteer/bin/python)
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

IFS=',' read -r -a ABL_LIST_RAW <<< "${ABLATIONS}"
ABL_LIST=()
for a in "${ABL_LIST_RAW[@]}"; do
  a="$(echo "${a}" | xargs)"
  [[ -n "${a}" ]] && ABL_LIST+=( "${a}" )
done
if [[ "${#ABL_LIST[@]}" -le 0 ]]; then
  echo "[error] Empty ABLATIONS list" >&2
  exit 2
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

# Load chosen params into maps keyed by "<model_basename>|<dataset>".
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

write_cfg() {
  local out_cfg="$1"
  local run_name="$2"
  local run_name_base="$3"
  local artifact_run_name_base="$4"
  local model_path="$5"
  local tp="$6"
  local max_num_seqs="$7"
  local dataset="$8"
  local prompt_template="$9"
  local train_split="${10}"
  local eval_split="${11}"
  local max_train="${12}"
  local max_eval="${13}"
  local tmax="${14}"
  local max_model_len="${15}"
  local offline_layer="${16:-}"
  local online_k_scale="${17:-}"
  local ablation="${18}"
  local stage_plan="${19}"
  local base_run_id="${20}"

  env BASE_CFG="${BASE_CFG}" OUT_CFG="${out_cfg}" RUN_NAME="${run_name}" RUN_NAME_BASE="${run_name_base}" \
    ARTIFACT_RUN_NAME_BASE="${artifact_run_name_base}" \
    MODEL_PATH="${model_path}" MODEL_TP="${tp}" MAX_NUM_SEQS="${max_num_seqs}" \
    DATASET="${dataset}" PROMPT_TEMPLATE="${prompt_template}" TRAIN_SPLIT="${train_split}" EVAL_SPLIT="${eval_split}" \
    MAX_TRAIN="${max_train}" MAX_EVAL="${max_eval}" TMAX="${tmax}" MAX_MODEL_LEN="${max_model_len}" \
    OFFLINE_LAYER="${offline_layer}" ONLINE_K_SCALE="${online_k_scale}" \
    RUN_GREEDY="${RUN_GREEDY}" RUN_ESM="${RUN_ESM}" \
    ABLATION="${ablation}" STAGE_PLAN="${stage_plan}" BASE_RUN_ID="${base_run_id}" OUTPUTS_ROOT="${OUTPUTS_ROOT}" \
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

max_train = _maybe_int(os.environ.get("MAX_TRAIN", "__KEEP__"))
max_eval = _maybe_int(os.environ.get("MAX_EVAL", "__KEEP__"))
if max_train != "__KEEP__":
    raw["task"]["max_train_examples"] = max_train
if max_eval != "__KEEP__":
    raw["task"]["max_eval_examples"] = max_eval

raw.setdefault("prompt", {})
raw["prompt"]["template"] = os.environ["PROMPT_TEMPLATE"]

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

abl = str(os.environ.get("ABLATION", "")).strip()
stage_plan = str(os.environ.get("STAGE_PLAN", "")).strip()

overrides = {}
if abl == "random_mining":
    overrides = {"offline_mine": {"method": "random"}}
elif abl == "top1_memory":
    overrides = {"offline_select": {"B": 1}, "online": {"k_retrieve": 1, "L": 1}}
elif abl == "use_random_memory":
    overrides = {"online": {"variant": "use_random_memory"}}
elif abl == "random_use_memory":
    overrides = {"online": {"variant": "random_use_memory"}}
elif abl == "no_probing":
    overrides = {"online": {"variant": "no_probing"}}
elif abl == "first_step_only":
    overrides = {"online": {"max_tool_m": 1}}
else:
    raise ValueError(f"Unknown ablation: {abl}")

def deep_merge(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst

deep_merge(raw, overrides)

# If no offline stages in plan, set eval.artifact_run_dir to base run.
stages_set = {s.strip() for s in stage_plan.split(",") if s.strip()}
if stages_set.isdisjoint({"mine", "select", "memory"}):
    raw.setdefault("eval", {})
    base_run_id = os.environ.get("BASE_RUN_ID", "latest")
    outputs_root = Path(os.environ.get("OUTPUTS_ROOT", "outputs"))
    run_name_base = os.environ.get("ARTIFACT_RUN_NAME_BASE") or os.environ.get("RUN_NAME_BASE", "")
    raw["eval"]["artifact_run_dir"] = str(outputs_root / run_name_base / base_run_id)

out_cfg.parent.mkdir(parents=True, exist_ok=True)
with out_cfg.open("w", encoding="utf-8") as f:
    yaml.safe_dump(raw, f, allow_unicode=True, sort_keys=False)
PY
}

jobs=()  # "IDX|RUN_NAME|ABLATION|CFG_PATH|CMD"
job_idx=0

if [[ "${LIST_JOBS}" == "1" ]]; then
  echo "[list_jobs] 1"
  echo "[ordering] model -> dataset -> ablation"
  echo
fi

for mspec in "${MODEL_SPECS[@]}"; do
  IFS='|' read -r model_key model_path model_tp model_max_num_seqs <<< "${mspec}"
  model_key_s="$(sanitize "${model_key}")"
  model_base="$(basename "${model_path}")"

  for dspec in "${DATASET_SPECS[@]}"; do
    IFS='|' read -r dataset prompt_template train_split eval_split max_train max_eval tmax max_model_len <<< "${dspec}"
    dataset_key="$(sanitize "${dataset}")"
    run_name_base="$(sanitize "${RUN_NAME_PREFIX}")_${model_key_s}_${dataset_key}"
    artifact_run_name_base="$(sanitize "${BASE_RUN_NAME_PREFIX}")_${model_key_s}_${dataset_key}"
    if [[ -z "${artifact_run_name_base}" ]]; then
      artifact_run_name_base="${run_name_base}"
    fi

    offline_layer=""
    online_k_scale=""
    if [[ "${USE_CHOSEN_PARAMS}" != "0" ]]; then
      param_key="${model_base}|${dataset}"
      offline_layer="${PARAM_LAYER_MAP["${param_key}"]:-}"
      online_k_scale="${PARAM_KSCALE_MAP["${param_key}"]:-}"
      if [[ "${LIST_JOBS}" != "1" && ( -z "${offline_layer}" || -z "${online_k_scale}" ) && "${missing_param_warned}" == "0" ]]; then
        echo "[warn] missing chosen params for some jobs (fallback to base config values)" >&2
        missing_param_warned=1
      fi
    fi

    for abl in "${ABL_LIST[@]}"; do
      abl_s="$(sanitize "${abl}")"
      if (( job_idx < RESUME_FROM )); then
        job_idx=$((job_idx + 1))
        continue
      fi

      default_eval_only=0
      stage_plan="${STAGES}"
      if [[ -z "${stage_plan}" ]]; then
        case "${abl_s}" in
          random_mining|top1_memory)
            stage_plan="mine,select,memory,eval"
            ;;
          *)
            stage_plan="eval"
            default_eval_only=1
            ;;
        esac
      fi

      base_run_id_resolved="${BASE_RUN_ID}"

      # If this ablation defaults to eval-only, try to find an existing artifact run.
      # Preference: BASE_RUN_ID (if not "latest") -> newest run_dir with memory/keys.npy.
      if [[ "${default_eval_only}" == "1" ]]; then
        artifact_root_base="${OUTPUTS_ROOT}/${artifact_run_name_base}"
        resolved=""

        if [[ -d "${artifact_root_base}" ]]; then
          if [[ "${BASE_RUN_ID}" != "latest" ]]; then
            expected_mem="${artifact_root_base}/${BASE_RUN_ID}/memory/keys.npy"
            if [[ -f "${expected_mem}" ]]; then
              resolved="${BASE_RUN_ID}"
            fi
          fi

          if [[ -z "${resolved}" ]]; then
            mapfile -t __artifact_dirs < <(find "${artifact_root_base}" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort -r)
            for cand in "${__artifact_dirs[@]}"; do
              [[ "${cand}" == "latest" ]] && continue
              expected_mem="${artifact_root_base}/${cand}/memory/keys.npy"
              if [[ -f "${expected_mem}" ]]; then
                resolved="${cand}"
                break
              fi
            done
          fi
        fi

        if [[ -n "${resolved}" ]]; then
          base_run_id_resolved="${resolved}"
        else
          echo "[info] artifact missing for eval-only ablation; switching to mine,select,memory,eval: ${artifact_root_base}/${BASE_RUN_ID}/memory/keys.npy" >&2
          stage_plan="mine,select,memory,eval"
        fi
      fi

      run_name="${run_name_base}_${abl_s}"
      cfg_path="${OUT_CFG_DIR}/${run_name}.yaml"

      if [[ "${LIST_JOBS}" == "1" ]]; then
        printf "%5d | model=%s | dataset=%s | ablation=%s | run_name=%s | run_id=%s | stages=%s\n" \
          "${job_idx}" "${model_key_s}" "${dataset_key}" "${abl_s}" "${run_name}" "${EXP_ID}" "${stage_plan}"
        job_idx=$((job_idx + 1))
        continue
      fi

      write_cfg \
        "${cfg_path}" "${run_name}" "${run_name_base}" "${artifact_run_name_base}" \
        "${model_path}" "${model_tp}" "${model_max_num_seqs}" \
        "${dataset}" "${prompt_template}" "${train_split}" "${eval_split}" \
        "${max_train}" "${max_eval}" "${tmax}" "${max_model_len}" \
        "${offline_layer}" "${online_k_scale}" "${abl_s}" "${stage_plan}" "${base_run_id_resolved}"

      cmd="cd \"${REPO_ROOT}\""
      IFS=',' read -r -a STAGE_LIST <<< "${stage_plan}"
      for st in "${STAGE_LIST[@]}"; do
        st="$(echo "${st}" | xargs)"
        [[ -z "${st}" ]] && continue
        cmd+=" && ${PY[*]} run.py --config \"${cfg_path}\" --run-id \"${EXP_ID}\" ${st}"
      done

      jobs+=( "${job_idx}|${run_name}|${abl_s}|${cfg_path}|${cmd}" )
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

echo "[mode] ablation experiments"
echo "[base_cfg] ${BASE_CFG}"
echo "[run_name_prefix] ${RUN_NAME_PREFIX}"
echo "[base_run_name_prefix] ${BASE_RUN_NAME_PREFIX}"
echo "[base_run_id_for_eval_only] ${BASE_RUN_ID}"
echo "[exp_id] ${EXP_ID}"
echo "[ablations] ${ABLATIONS}"
echo "[gpus] ${GPUS}"
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
    IFS='|' read -r idx rn abl cfg cmd <<< "${j}"
    echo "[dry_run] idx=${idx} ablation=${abl} cfg=${cfg} cmd=${cmd}"
  done
  exit 0
fi

free_gpus=( "${GPU_LIST[@]}" )
declare -A pid_to_gpu=()
declare -A pid_to_name=()
failed=0
failed_jobs=()

EVENT_TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ablation_experiments.${EXP_ID}.XXXXXX")"
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
}
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM
trap cleanup_events EXIT

launch_job() {
  local job="$1"
  local gpu="$2"
  IFS='|' read -r idx rn abl cfg cmd <<< "${job}"
  local log_path=""
  if [[ -n "${LOG_DIR}" ]]; then
    log_path="${LOG_DIR}/${rn}_${abl}_${EXP_ID}.log"
  else
    local run_dir="${OUTPUTS_ROOT}/${rn}/${EXP_ID}"
    local run_logs="${run_dir}/logs"
    mkdir -p "${run_logs}"
    log_path="${run_logs}/stdout.log"
  fi

  echo "[start] idx=${idx} gpu=${gpu} run=${rn}"
  echo "  cfg: ${cfg}"
  echo "  log: ${log_path}"

  (
    set +e
    export CUDA_VISIBLE_DEVICES="${gpu}"
    bash -lc "${cmd}"
    code=$?
    echo "${BASHPID}:${code}" > "${EVENT_FIFO}"
    exit "${code}"
  ) >"${log_path}" 2>&1 &
  local pid=$!
  pid_to_gpu["${pid}"]="${gpu}"
  pid_to_name["${pid}"]="${rn}"
}

job_ptr=0
running=0

while (( job_ptr < ${#jobs[@]} )) || (( running > 0 )); do
  while (( job_ptr < ${#jobs[@]} )) && (( running < ${#GPU_LIST[@]} )) && (( ${#free_gpus[@]} > 0 )); do
    next_gpu="${free_gpus[0]}"
    free_gpus=( "${free_gpus[@]:1}" )
    launch_job "${jobs[$job_ptr]}" "${next_gpu}"
    job_ptr=$((job_ptr + 1))
    running=$((running + 1))
  done

  if (( running > 0 )); then
    read -r finished_record < "${EVENT_FIFO}"
    if [[ -n "${finished_record}" ]]; then
      finished_pid="${finished_record%%:*}"
      finished_code="${finished_record##*:}"
      if [[ "${finished_code}" != "0" ]]; then
        failed=$((failed + 1))
        failed_jobs+=( "${pid_to_name["${finished_pid}"]}" )
        echo "[fail] ${pid_to_name["${finished_pid}"]}" "(exit=${finished_code})" >&2
      fi
      free_gpus+=( "${pid_to_gpu["${finished_pid}"]}" )
      unset pid_to_gpu["${finished_pid}"]
      unset pid_to_name["${finished_pid}"]
      running=$((running - 1))
    fi
  fi
done

if (( failed > 0 )); then
  echo "[done] ${failed} job(s) failed:" >&2
  for j in "${failed_jobs[@]}"; do
    echo "  - ${j}" >&2
  done
  exit 1
fi

echo "[done] all jobs finished (n=${total_jobs})"
