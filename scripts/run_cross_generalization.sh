#!/usr/bin/env bash
set -euo pipefail

# Cross-dataset generalization runner.
# 1) Uses high-tier params from analysis/chosen_params_for_main.csv.
# 2) Runs offline (mine/select/memory) once per dataset for the chosen model.
# 3) Reuses each dataset's memory to run online eval on every other dataset.
#
# Quickstart:
#   GPU=0 bash ./scripts/run_cross_generalization.sh
#   PREP_WITH_MAIN=0 GPU=0 bash ./scripts/run_cross_generalization.sh
#   SKIP_OFFLINE=1 ARTIFACT_RUN_ID=latest GPU=0 bash ./scripts/run_cross_generalization.sh

# =========================
# User-editable (top)
# =========================
# <model_key>|<name_or_path>|<tensor_parallel_size>|<max_num_seqs>
MODEL_SPEC="${MODEL_SPEC:-ds_r1_qwen_7b|huggingface_models/DeepSeek-R1-Distill-Qwen-7B|1|256}"

# Base config template.
BASE_CFG="${BASE_CFG:-configs/default}"

# Param source (tier=high by default).
CHOSEN_PARAMS_CSV="${CHOSEN_PARAMS_CSV:-analysis/chosen_params_for_main.csv}"
CHOSEN_PARAMS_TIER="${CHOSEN_PARAMS_TIER:-high}"

# Run naming + IDs.
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-generalization}"
EXP_ID="${EXP_ID:-$(date +%Y%m%d_%H%M%S)}"
# Which offline run_id to use when loading memory for eval (default: same as offline run_id).
ARTIFACT_RUN_ID_WAS_SET=0
if [[ -n "${ARTIFACT_RUN_ID+x}" ]]; then
  ARTIFACT_RUN_ID_WAS_SET=1
fi
ARTIFACT_RUN_ID="${ARTIFACT_RUN_ID:-${EXP_ID}}"

# Runtime controls.
GPU="${GPU:-0,1,2,3,4,5,6,7}"
RUN_GREEDY="${RUN_GREEDY:-0}"
RUN_ESM="${RUN_ESM:-1}"
OFFLINE_STAGES="${OFFLINE_STAGES:-mine,select,memory}"
INCLUDE_SELF="${INCLUDE_SELF:-0}"   # Set to 1 to also evaluate src->src.
SKIP_OFFLINE="${SKIP_OFFLINE:-0}"   # Set to 1 to reuse existing offline artifacts.
DRY_RUN="${DRY_RUN:-0}"
OUT_CFG_DIR="${OUT_CFG_DIR:-configs/_generalization_generated/${EXP_ID}}"

# Pre-run main_experiments to regenerate memories for the chosen model/datasets.
PREP_WITH_MAIN="${PREP_WITH_MAIN:-1}"
USE_MAIN_ARTIFACTS="${USE_MAIN_ARTIFACTS:-${PREP_WITH_MAIN}}"
MAIN_RUN_NAME_PREFIX="${MAIN_RUN_NAME_PREFIX:-main}"
MAIN_STAGES="${MAIN_STAGES:-mine,select,memory}"
MAIN_EXP_ID="${MAIN_EXP_ID:-${EXP_ID}}"
MAIN_MODEL_SPEC="${MAIN_MODEL_SPEC:-${MODEL_SPEC}}"
MAIN_GPUS="${MAIN_GPUS:-${GPU}}"
MAIN_SCRIPT="${MAIN_SCRIPT:-scripts/run_main_experiments.sh}"

# Optional: force conda.
USE_CONDA_RUN="${USE_CONDA_RUN:-0}"
CONDA_ENV="${CONDA_ENV:-easysteer}"

# Dataset specs: <dataset>|<prompt_template>|<train_split>|<eval_split>|<max_train>|<max_eval>|<T_max>|<max_model_len>
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

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${MAIN_SCRIPT}" != /* ]]; then
  MAIN_SCRIPT="${REPO_ROOT}/${MAIN_SCRIPT}"
fi

PY=(python)
if [[ "${USE_CONDA_RUN}" == "1" ]]; then
  PY=(conda run -n "${CONDA_ENV}" python)
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

f_tag() {
  # Convert numeric/rational tokens to tag-safe form (aligns with run_grid_sweep.sh).
  local s="$1"
  s="${s//-/m}"
  s="${s//./p}"
  s="${s//\//d}"
  echo "${s}"
}

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

if [[ ! -f "${CHOSEN_PARAMS_CSV}" ]]; then
  echo "[error] CHOSEN_PARAMS_CSV not found: ${CHOSEN_PARAMS_CSV}" >&2
  exit 2
fi

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

declare -A DATASET_SPEC_MAP
for dspec in "${DATASET_SPECS[@]}"; do
  IFS='|' read -r ds _ <<< "${dspec}"
  DATASET_SPEC_MAP["${ds}"]="${dspec}"
done

IFS='|' read -r MODEL_KEY MODEL_PATH MODEL_TP MODEL_MAX_NUM_SEQS <<< "${MODEL_SPEC}"
if [[ -z "${MODEL_PATH}" ]]; then
  echo "[error] MODEL_SPEC is empty; expected <key>|<path>|<tp>|<max_num_seqs>" >&2
  exit 2
fi
MODEL_KEY="$(sanitize "${MODEL_KEY:-${MODEL_PATH}}")"
MODEL_BASENAME="$(basename "${MODEL_PATH}")"
MODEL_TP="${MODEL_TP:-1}"
MODEL_MAX_NUM_SEQS="${MODEL_MAX_NUM_SEQS:-__KEEP__}"
IFS='|' read -r MAIN_MODEL_KEY MAIN_MODEL_PATH MAIN_MODEL_TP MAIN_MODEL_MAX_NUM_SEQS <<< "${MAIN_MODEL_SPEC}"
if [[ -z "${MAIN_MODEL_PATH}" ]]; then
  MAIN_MODEL_PATH="${MODEL_PATH}"
fi
if [[ -z "${MAIN_MODEL_KEY}" ]]; then
  MAIN_MODEL_KEY="${MODEL_KEY}"
fi
MAIN_MODEL_KEY="$(sanitize "${MAIN_MODEL_KEY:-${MAIN_MODEL_PATH}}")"
if [[ "${USE_MAIN_ARTIFACTS}" == "1" && "${ARTIFACT_RUN_ID_WAS_SET}" == "0" ]]; then
  ARTIFACT_RUN_ID="${MAIN_EXP_ID}"
fi

METHODS=()
if [[ "${RUN_GREEDY}" != "0" ]]; then
  METHODS+=( "greedy" )
fi
if [[ "${RUN_ESM}" != "0" ]]; then
  METHODS+=( "esm" )
fi
if (( ${#METHODS[@]} == 0 )); then
  echo "[error] both RUN_GREEDY and RUN_ESM are disabled; nothing to run" >&2
  exit 2
fi
METHODS_CSV="$(IFS=','; echo "${METHODS[*]}")"

mkdir -p "${OUT_CFG_DIR}"
if [[ "${USE_MAIN_ARTIFACTS}" == "1" && "${SKIP_OFFLINE}" != "1" ]]; then
  echo "[info] USE_MAIN_ARTIFACTS=1 -> skipping offline stages in this script (memories will come from run_main_experiments)" >&2
  SKIP_OFFLINE=1
fi

mapfile -t __param_rows < <(CHOSEN_PARAMS_CSV="${CHOSEN_PARAMS_CSV}" MODEL_NAME="${MODEL_BASENAME}" CHOSEN_PARAMS_TIER="${CHOSEN_PARAMS_TIER}" "${PY[@]}" - <<'PY'
import os
from pathlib import Path

import pandas as pd

csv_path = Path(os.environ["CHOSEN_PARAMS_CSV"])
model = str(os.environ["MODEL_NAME"]).strip()
tier = str(os.environ["CHOSEN_PARAMS_TIER"]).strip()
df = pd.read_csv(csv_path)
df = df[df["tier"] == tier]
df["model_key"] = df["models"].astype(str).str.replace("\\\\", "/").str.split("/").str[-1]
df = df[df["model_key"] == model]
if df.empty:
    raise SystemExit(f"No rows found in {csv_path} for model={model} tier={tier}")
for _, row in df.iterrows():
    ds = str(row["dataset"]).strip()
    l = str(row["offline_candidate_layers"]).strip()
    k = str(row["online_k_scale"]).strip()
    print(f"{ds}|{l}|{k}")
PY
)

declare -A PARAM_LAYER_MAP
declare -A PARAM_KSCALE_MAP
declare -A PARAM_SEEN
for line in "${__param_rows[@]}"; do
  IFS='|' read -r ds l k <<< "${line}"
  PARAM_LAYER_MAP["${ds}"]="${l}"
  PARAM_KSCALE_MAP["${ds}"]="${k}"
  PARAM_SEEN["${ds}"]=1
done

DATASET_LIST=()
for dspec in "${DATASET_SPECS[@]}"; do
  IFS='|' read -r ds _ <<< "${dspec}"
  if [[ -n "${PARAM_SEEN[${ds}]:-}" ]]; then
    DATASET_LIST+=( "${ds}" )
  fi
done

for ds in "${!PARAM_SEEN[@]}"; do
  if [[ -z "${DATASET_SPEC_MAP[${ds}]+x}" ]]; then
    echo "[warn] missing dataset spec for ${ds}; it will be skipped" >&2
  fi
done

if (( ${#DATASET_LIST[@]} == 0 )); then
  echo "[error] no datasets to run after intersecting chosen_params with DATASET_SPECS" >&2
  exit 2
fi

# Validate grid outputs contain the chosen (layer, k_scale) combos for all datasets.
missing_grid=()
grid_choice=()
for ds in "${DATASET_LIST[@]}"; do
  ds_key="$(sanitize "${ds}")"
  grid_root="${OUTPUTS_ROOT}/grid_${MODEL_KEY}_${ds_key}"
  offline_layer="${PARAM_LAYER_MAP[${ds}]:-}"
  online_k_scale="${PARAM_KSCALE_MAP[${ds}]:-}"
  if [[ -z "${offline_layer}" || -z "${online_k_scale}" ]]; then
    missing_grid+=( "${ds}|${grid_root}|missing chosen params (L=?, ks=?)" )
    continue
  fi
  if [[ ! -d "${grid_root}" ]]; then
    missing_grid+=( "${ds}|${grid_root}|grid root not found (need L=${offline_layer}, ks=${online_k_scale})" )
    continue
  fi

  lt="$(f_tag "${offline_layer}")"
  kt="$(f_tag "${online_k_scale}")"
  shopt -s nullglob
  cand=( "${grid_root}"/*__L${lt}_eta*_ks${kt} )
  shopt -u nullglob
  if (( ${#cand[@]} == 0 )); then
    missing_grid+=( "${ds}|${grid_root}|no run matching L=${offline_layer}, ks=${online_k_scale}" )
    continue
  fi
  # Pick lexicographically newest (run_ids start with timestamp).
  newest=""
  for d in "${cand[@]}"; do
    if [[ -z "${newest}" || "${d}" > "${newest}" ]]; then
      newest="${d}"
    fi
  done
  if [[ ! -f "${newest}/tables/main_results_single.csv" ]]; then
    missing_grid+=( "${ds}|${newest}|missing tables/main_results_single.csv" )
    continue
  fi
  grid_choice+=( "${ds}|${newest}" )
done

if (( ${#missing_grid[@]} > 0 )); then
  echo "[error] missing grid outputs for required params (stop):" >&2
  for entry in "${missing_grid[@]}"; do
    IFS='|' read -r ds root reason <<< "${entry}"
    echo "  - dataset=${ds} root=${root} reason=${reason}" >&2
  done
  exit 2
fi

echo "[grid] using latest sweep runs:"
for entry in "${grid_choice[@]}"; do
  IFS='|' read -r ds dir <<< "${entry}"
  echo "  - ${ds}: ${dir}"
done
echo

if [[ "${PREP_WITH_MAIN}" == "1" ]]; then
  if [[ "${USE_MAIN_ARTIFACTS}" != "1" ]]; then
    echo "[warn] PREP_WITH_MAIN=1 but USE_MAIN_ARTIFACTS=0; main run will not be reused" >&2
  fi
  if [[ ! -x "${MAIN_SCRIPT}" ]]; then
    echo "[error] MAIN_SCRIPT not found or not executable: ${MAIN_SCRIPT}" >&2
    exit 2
  fi

  dataset_override="$(printf "%s\n" "${DATASET_SPECS[@]}")"
  main_model_override_escaped="$(printf "%q" "${MAIN_MODEL_SPEC}")"
  dataset_override_escaped="$(printf "%q" "${dataset_override}")"
  main_gpus_escaped="$(printf "%q" "${MAIN_GPUS}")"
  main_stages_escaped="$(printf "%q" "${MAIN_STAGES}")"
  main_run_prefix_escaped="$(printf "%q" "${MAIN_RUN_NAME_PREFIX}")"
  main_exp_id_escaped="$(printf "%q" "${MAIN_EXP_ID}")"
  base_cfg_escaped="$(printf "%q" "${BASE_CFG}")"
  chosen_params_csv_escaped="$(printf "%q" "${CHOSEN_PARAMS_CSV}")"
  chosen_params_tier_escaped="$(printf "%q" "${CHOSEN_PARAMS_TIER}")"
  run_greedy_escaped="$(printf "%q" "${RUN_GREEDY}")"
  run_esm_escaped="$(printf "%q" "${RUN_ESM}")"
  use_conda_run_escaped="$(printf "%q" "${USE_CONDA_RUN}")"
  conda_env_escaped="$(printf "%q" "${CONDA_ENV}")"

  main_cmd="cd \"${REPO_ROOT}\""
  main_cmd+=" && MODEL_SPECS_OVERRIDE=${main_model_override_escaped}"
  main_cmd+=" DATASET_SPECS_OVERRIDE=${dataset_override_escaped}"
  main_cmd+=" GPUS=${main_gpus_escaped}"
  main_cmd+=" STAGES=${main_stages_escaped}"
  main_cmd+=" RUN_NAME_PREFIX=${main_run_prefix_escaped}"
  main_cmd+=" EXP_ID=${main_exp_id_escaped}"
  main_cmd+=" BASE_CFG=${base_cfg_escaped}"
  main_cmd+=" CHOSEN_PARAMS_CSV=${chosen_params_csv_escaped}"
  main_cmd+=" CHOSEN_PARAMS_TIER=${chosen_params_tier_escaped}"
  main_cmd+=" RUN_GREEDY=${run_greedy_escaped}"
  main_cmd+=" RUN_ESM=${run_esm_escaped}"
  main_cmd+=" USE_CONDA_RUN=${use_conda_run_escaped}"
  main_cmd+=" CONDA_ENV=${conda_env_escaped}"
  main_cmd+=" bash \"${MAIN_SCRIPT}\""

  echo "[prep] running run_main_experiments to regenerate memories (exp_id=${MAIN_EXP_ID}, run_prefix=${MAIN_RUN_NAME_PREFIX})"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run][prep] ${main_cmd}"
  else
    bash -lc "${main_cmd}"
  fi
fi

write_cfg() {
  local out_cfg="$1"
  local run_name="$2"
  local dataset="$3"
  local prompt_template="$4"
  local train_split="$5"
  local eval_split="$6"
  local max_train="$7"
  local max_eval="$8"
  local tmax="$9"
  local max_model_len="${10}"
  local offline_layer="${11:-}"
  local online_k_scale="${12:-}"
  local artifact_dir="${13:-}"

  env BASE_CFG="${BASE_CFG}" OUT_CFG="${out_cfg}" RUN_NAME="${run_name}" \
    MODEL_PATH="${MODEL_PATH}" MODEL_TP="${MODEL_TP}" \
    MAX_NUM_SEQS="${MODEL_MAX_NUM_SEQS}" \
    DATASET="${dataset}" PROMPT_TEMPLATE="${prompt_template}" TRAIN_SPLIT="${train_split}" EVAL_SPLIT="${eval_split}" \
    MAX_TRAIN="${max_train}" MAX_EVAL="${max_eval}" TMAX="${tmax}" MAX_MODEL_LEN="${max_model_len}" \
    OFFLINE_LAYER="${offline_layer}" ONLINE_K_SCALE="${online_k_scale}" \
    ARTIFACT_DIR="${artifact_dir}" METHODS_CSV="${METHODS_CSV}" \
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

def _maybe_int(s: str):
    s = (s or "").strip()
    if s == "" or s == "__KEEP__":
        return "__KEEP__"
    if s.lower() in {"none", "null"}:
        return None
    return int(s)

raw.setdefault("outputs", {})
raw["outputs"]["run_name"] = os.environ["RUN_NAME"]

raw.setdefault("model", {})
raw["model"]["name_or_path"] = os.environ["MODEL_PATH"]
tp_s = str(os.environ.get("MODEL_TP", "__KEEP__")).strip()
if tp_s and tp_s != "__KEEP__":
    raw["model"]["tensor_parallel_size"] = int(tp_s)

max_num_seqs = _maybe_int(os.environ.get("MAX_NUM_SEQS", "__KEEP__"))
if max_num_seqs != "__KEEP__":
    if max_num_seqs is None:
        raise ValueError("MAX_NUM_SEQS cannot be null/none")
    raw["model"]["max_num_seqs"] = int(max_num_seqs)

raw.setdefault("task", {})
raw["task"]["dataset"] = os.environ["DATASET"]
raw["task"]["train_split"] = os.environ["TRAIN_SPLIT"]
raw["task"]["eval_split"] = os.environ["EVAL_SPLIT"]

ds_train = _maybe_int(os.environ.get("MAX_TRAIN", "__KEEP__"))
ds_eval = _maybe_int(os.environ.get("MAX_EVAL", "__KEEP__"))
if ds_train != "__KEEP__":
    raw["task"]["max_train_examples"] = ds_train
if ds_eval != "__KEEP__":
    raw["task"]["max_eval_examples"] = ds_eval

raw.setdefault("prompt", {})
raw["prompt"]["template"] = os.environ["PROMPT_TEMPLATE"]

raw.setdefault("decode", {})
tmax = _maybe_int(os.environ.get("TMAX", "__KEEP__"))
if tmax != "__KEEP__":
    raw["decode"]["max_new_tokens"] = tmax
    raw.setdefault("offline_mine", {})
    raw["offline_mine"]["max_new_tokens"] = tmax

max_model_len = _maybe_int(os.environ.get("MAX_MODEL_LEN", "__KEEP__"))
if max_model_len != "__KEEP__":
    raw.setdefault("model", {})
    raw["model"]["max_model_len"] = max_model_len

offline_layer = str(os.environ.get("OFFLINE_LAYER", "")).strip()
if offline_layer != "":
    raw.setdefault("offline_mine", {})
    raw["offline_mine"]["candidate_layers"] = [offline_layer]

online_k = str(os.environ.get("ONLINE_K_SCALE", "")).strip()
if online_k != "":
    raw.setdefault("online", {})
    raw["online"]["k_scale"] = float(online_k)

methods = [m.strip() for m in str(os.environ.get("METHODS_CSV", "")).split(",") if m.strip() != ""]
if not methods:
    methods = ["esm"]
raw.setdefault("eval", {})
raw["eval"]["methods"] = methods
raw["eval"]["ablations"] = []

artifact_dir = str(os.environ.get("ARTIFACT_DIR", "")).strip()
if artifact_dir != "":
    raw["eval"]["artifact_run_dir"] = artifact_dir

out_cfg.parent.mkdir(parents=True, exist_ok=True)
with out_cfg.open("w", encoding="utf-8") as f:
    yaml.safe_dump(raw, f, allow_unicode=True, sort_keys=False)
PY
}

echo "[mode] cross generalization"
echo "[model] ${MODEL_PATH} (key=${MODEL_KEY}, tp=${MODEL_TP}, max_num_seqs=${MODEL_MAX_NUM_SEQS})"
echo "[datasets] ${DATASET_LIST[*]}"
echo "[main prep] prep_with_main=${PREP_WITH_MAIN} use_main_artifacts=${USE_MAIN_ARTIFACTS} main_exp_id=${MAIN_EXP_ID} main_run_prefix=${MAIN_RUN_NAME_PREFIX}"
echo "[base_cfg] ${BASE_CFG}"
echo "[exp_id / offline run_id] ${EXP_ID}"
echo "[artifact_run_id for eval] ${ARTIFACT_RUN_ID}"
echo "[outputs_root] ${OUTPUTS_ROOT}"
echo "[gpu] ${GPU}"
echo "[methods] ${METHODS_CSV}"
echo

IFS=',' read -r -a OFFLINE_STAGE_LIST <<< "${OFFLINE_STAGES}"

declare -A OFFLINE_RUN_MAP

# 1) Offline for each dataset (mine/select/memory)
for ds in "${DATASET_LIST[@]}"; do
  spec="${DATASET_SPEC_MAP[${ds}]}"
  if [[ -z "${spec}" ]]; then
    echo "[skip] no dataset spec for ${ds}" >&2
    continue
  fi
  IFS='|' read -r dataset prompt_template train_split eval_split max_train max_eval tmax max_model_len <<< "${spec}"
  offline_layer="${PARAM_LAYER_MAP[${dataset}]:-}"
  online_k_scale="${PARAM_KSCALE_MAP[${dataset}]:-}"

  if [[ -z "${offline_layer}" || -z "${online_k_scale}" ]]; then
    echo "[warn] missing chosen params for ${dataset}; using base config values" >&2
  fi

  ds_key="$(sanitize "${dataset}")"
  if [[ "${USE_MAIN_ARTIFACTS}" == "1" ]]; then
    run_name="${MAIN_RUN_NAME_PREFIX}_${MAIN_MODEL_KEY}_${ds_key}"
    OFFLINE_RUN_MAP["${dataset}"]="${run_name}"
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "[dry-run][offline-reuse] ${dataset}: use ${run_name} (artifact_run_id=${ARTIFACT_RUN_ID})"
    else
      echo "[reuse offline] ${dataset}: ${run_name} (artifact_run_id=${ARTIFACT_RUN_ID})"
    fi
    continue
  fi

  run_name="${RUN_NAME_PREFIX}_${MODEL_KEY}_offline_${ds_key}"
  cfg_path="${OUT_CFG_DIR}/${run_name}.yaml"
  write_cfg "${cfg_path}" "${run_name}" "${dataset}" "${prompt_template}" "${train_split}" "${eval_split}" \
    "${max_train}" "${max_eval}" "${tmax}" "${max_model_len}" \
    "${offline_layer}" "${online_k_scale}" ""

  OFFLINE_RUN_MAP["${dataset}"]="${run_name}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run][offline] cfg=${cfg_path}"
    continue
  fi

  if [[ "${SKIP_OFFLINE}" != "0" ]]; then
    echo "[skip offline] ${run_name}"
    continue
  fi

  cmd="cd \"${REPO_ROOT}\""
  for st in "${OFFLINE_STAGE_LIST[@]}"; do
    st="$(echo "${st}" | xargs)"
    if [[ -z "${st}" ]]; then
      continue
    fi
    cmd+=" && ${PY[*]} run.py --config \"${cfg_path}\" --run-id \"${EXP_ID}\" ${st}"
  done

  echo "[run][offline] ${run_name}"
  CUDA_VISIBLE_DEVICES="${GPU}" bash -lc "${cmd}"
done

# 2) Cross-eval: reuse each memory on every target dataset.
for src_ds in "${DATASET_LIST[@]}"; do
  src_run="${OFFLINE_RUN_MAP[${src_ds}]}"
  if [[ -z "${src_run}" ]]; then
    echo "[skip cross] missing offline run for ${src_ds}" >&2
    continue
  fi
  src_art_dir="${OUTPUTS_ROOT}/${src_run}/${ARTIFACT_RUN_ID}"

  for tgt_ds in "${DATASET_LIST[@]}"; do
    if [[ "${INCLUDE_SELF}" == "0" && "${src_ds}" == "${tgt_ds}" ]]; then
      continue
    fi
    spec="${DATASET_SPEC_MAP[${tgt_ds}]}"
    if [[ -z "${spec}" ]]; then
      echo "[skip] no dataset spec for target ${tgt_ds}" >&2
      continue
    fi
    IFS='|' read -r dataset prompt_template train_split eval_split max_train max_eval tmax max_model_len <<< "${spec}"
    offline_layer="${PARAM_LAYER_MAP[${dataset}]:-}"
    online_k_scale="${PARAM_KSCALE_MAP[${dataset}]:-}"

    if [[ -z "${offline_layer}" || -z "${online_k_scale}" ]]; then
      echo "[warn] missing chosen params for target ${dataset}; using base config values" >&2
    fi

    src_key="$(sanitize "${src_ds}")"
    tgt_key="$(sanitize "${dataset}")"
    run_name="${RUN_NAME_PREFIX}_${MODEL_KEY}_mem_${src_key}_on_${tgt_key}"
    cfg_path="${OUT_CFG_DIR}/${run_name}.yaml"
    write_cfg "${cfg_path}" "${run_name}" "${dataset}" "${prompt_template}" "${train_split}" "${eval_split}" \
      "${max_train}" "${max_eval}" "${tmax}" "${max_model_len}" \
      "${offline_layer}" "${online_k_scale}" "${src_art_dir}"

    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "[dry-run][eval] ${src_ds} -> ${tgt_ds} cfg=${cfg_path} artifact=${src_art_dir}"
      continue
    fi

    cmd="cd \"${REPO_ROOT}\" && ${PY[*]} run.py --config \"${cfg_path}\" --run-id \"${EXP_ID}\" eval"
    echo "[run][eval] mem=${src_ds} -> tgt=${tgt_ds}"
    CUDA_VISIBLE_DEVICES="${GPU}" bash -lc "${cmd}"
  done
done

echo "[done] exp_id=${EXP_ID}"
