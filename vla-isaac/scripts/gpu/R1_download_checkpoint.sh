#!/usr/bin/env bash
# R1. Download the pilot checkpoint (docs/PILOT_AND_ORDER.md R1), pinned to the revision it reads first.
# Settles V-hf-rev (the hash), V-ckpt-cfg and V-stats-format (the files and their shapes), and writes the
# training-spec JSON. A gated repo needs HF_TOKEN exported in your shell; this script never prints it.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

require_env ARENA_ROOT MODELS_DIR
require_vla
hf="$ARENA_ROOT/submodules/Isaac-GR00T/.venv/bin/hf"
require_file "$hf" "run R0_setup.sh (the hf CLI comes with GR00T's huggingface_hub)"
[[ -n "${HF_TOKEN:-}" ]] || echo "note: HF_TOKEN is not set; fine for a public repo, a gated one will refuse"
begin_measurement

repo="$(rb_get checkpoint.repo)"
sub="$(rb_get checkpoint.subfolder)"
ckpt="$MODELS_DIR/ckpt"

sha="$("$(gr00t_py)" -c 'import sys; from huggingface_hub import HfApi; print(HfApi().model_info(sys.argv[1]).sha)' "$repo")"
echo "$repo @ $sha"
echo "$sha" > "$out/revision.txt"

if [[ -f "$ckpt/config.json" && -f "$MODELS_DIR/ckpt.revision" && "$(cat "$MODELS_DIR/ckpt.revision")" == "$sha" ]]; then
  echo "checkpoint already at $ckpt (revision $sha)"
else
  mkdir -p "$MODELS_DIR"
  "$hf" download "$repo" --include "$sub/*" --repo-type model --revision "$sha" --local-dir "$MODELS_DIR/_dl" \
    2>&1 | tee "$out/download.log"
  rm -rf "$ckpt.partial"
  cp -r "$MODELS_DIR/_dl/$sub" "$ckpt.partial"
  rm -rf "$ckpt"
  mv "$ckpt.partial" "$ckpt"
  echo "$sha" > "$MODELS_DIR/ckpt.revision"
fi

grep -n "tune_top_llm_layers\|num_inference_timesteps\|action_horizon" "$ckpt/config.json" | tee "$out/config_grep.txt" || true
tool checkpoint-report --ckpt "$ckpt" --json-out "$out/checkpoint.json"
"$(vla_py)" -m vla_isaac.specs.extract_training_spec --backend "$VLA_BACKEND_CONFIG" --checkpoint-dir "$ckpt" \
  --output "$out/training_spec.json" 2>&1 | tee "$out/training_spec.log"

finish_measurement --json "checkpoint=$out/checkpoint.json" --set "revision_sha=$sha" --set "repo=$repo" \
  --set "training_spec=$out/training_spec.json"
echo "pin it: policy-wrapper/config/eval.yaml server.checkpoint_revision and finetune.base_revision = $sha"
