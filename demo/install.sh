#!/usr/bin/env bash
# Install dependencies and pre-download the models for the persona-vectors LSH demo.
#
# Assumes the `persona-lsh` conda env from the repo README already exists and
# has python, numpy, torch, and pytest installed. This script layers
# `transformers` + `accelerate` on top and pre-fetches the demo models.
#
# Default models (heavyweight; ~30 GB combined disk):
#   - Qwen/Qwen2.5-7B-Instruct          (open)
#   - meta-llama/Llama-3.1-8B-Instruct  (gated: requires HF access request)
#
# Override via env var, comma-separated:
#   DEMO_MODELS="Qwen/Qwen2.5-1.5B-Instruct"  bash demo/install.sh
#   PERSONA_LSH_ENV=persona-lsh DEMO_MODELS="Qwen/Qwen2.5-7B-Instruct,mistralai/Mistral-7B-Instruct-v0.3" bash demo/install.sh
set -euo pipefail

ENV_NAME="${PERSONA_LSH_ENV:-persona-lsh}"
DEMO_MODELS_DEFAULT="Qwen/Qwen2.5-7B-Instruct,meta-llama/Llama-3.1-8B-Instruct"
DEMO_MODELS="${DEMO_MODELS:-${DEMO_MODELS_DEFAULT}}"

# Resolve the env's python without requiring `conda activate`.
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
else
    CONDA_BASE="${HOME}/anaconda3"
fi
PY="${CONDA_BASE}/envs/${ENV_NAME}/bin/python"

if [ ! -x "${PY}" ]; then
    echo "ERROR: conda env '${ENV_NAME}' python not found at ${PY}" >&2
    echo "Create it first per the repo README, e.g.:" >&2
    echo "  conda create -n ${ENV_NAME} python=3.12 pip" >&2
    echo "  ${PY} -m pip install numpy pytest pandas" >&2
    echo "  ${PY} -m pip install torch --index-url https://download.pytorch.org/whl/cpu" >&2
    exit 1
fi

echo "[install] using python: ${PY}"
"${PY}" --version

echo "[install] adding transformers + accelerate + safetensors + openai..."
"${PY}" -m pip install --quiet transformers accelerate safetensors huggingface_hub openai

# Warn early about the Llama gated repo.
if [[ "${DEMO_MODELS}" == *meta-llama* ]]; then
    cat <<'EOF'

[install] NOTE: meta-llama/* repos are gated on HuggingFace.
  If snapshot_download fails with 401/403:
    1. Visit https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct and click
       "Request access" (usually approved within hours).
    2. Run:  huggingface-cli login   (paste a read token from
             https://huggingface.co/settings/tokens)
    3. Re-run this script.

EOF
fi

IFS=',' read -ra MODELS <<< "${DEMO_MODELS}"
for MODEL in "${MODELS[@]}"; do
    MODEL="$(echo "$MODEL" | xargs)"  # trim whitespace
    [ -z "$MODEL" ] && continue
    echo "[install] pre-downloading ${MODEL} ..."
    "${PY}" - <<PYEOF || { echo "[install] download of ${MODEL} failed; see message above" >&2; exit 1; }
from huggingface_hub import snapshot_download
path = snapshot_download("${MODEL}")
print(f"[install] cached at: {path}")
PYEOF
done

echo "[install] done. Run the demo with:"
echo "  ${PY} demo/run_demo.py"
