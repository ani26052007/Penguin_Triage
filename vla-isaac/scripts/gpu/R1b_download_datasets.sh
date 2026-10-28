#!/usr/bin/env bash
# R1b. Download the original LeRobot dataset (its meta/tasks.jsonl: V-train-instr; the nominal half of the
# training mix) and the annotated Mimic source HDF5 (V-mimic-source), then link both where the backend
# config's datasets map points. The LeRobot directory inside the repo is unverified: the script lists the
# repo and picks the one directory with meta/tasks.jsonl whose path contains the runbook's hint, or stops
# and asks for LEROBOT_PREFIX.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

require_env ARENA_ROOT DATA_DIR
require_vla
hf="$ARENA_ROOT/submodules/Isaac-GR00T/.venv/bin/hf"
require_file "$hf" "run R0_setup.sh"
[[ -n "${HF_TOKEN:-}" ]] || echo "note: HF_TOKEN is not set; fine for a public repo, a gated one will refuse"
begin_measurement

repo="$(rb_get dataset.repo)"
rev="$(rb_get dataset.revision)"
mimic_src="$(rb_get dataset.mimic_source)"
dl="$DATA_DIR/hf_dataset"
mapfile -t targets < <(tool dataset-paths --runbook "$runbook_cfg" --backend "$VLA_BACKEND_CONFIG")
((${#targets[@]} == 2)) || die "could not read the dataset paths from $VLA_BACKEND_CONFIG"
lerobot_link="${targets[0]}"
mimic_link="${targets[1]}"

"$(gr00t_py)" -c 'import sys; from huggingface_hub import list_repo_files as l
print("\n".join(l(sys.argv[1], repo_type="dataset", revision=sys.argv[2])))' "$repo" "$rev" > "$out/repo_files.txt"
echo "$(wc -l < "$out/repo_files.txt") files in $repo@$rev"

prefix="${LEROBOT_PREFIX:-$(tool pick-lerobot --files "$out/repo_files.txt" --hint "$(rb_get dataset.lerobot_hint)")}"
echo "LeRobot directory: $prefix"
echo "$prefix" > "$out/lerobot_prefix.txt"

mkdir -p "$dl"
if [[ ! -f "$dl/$prefix/meta/tasks.jsonl" ]]; then
  "$hf" download "$repo" --include "$prefix/*" --repo-type dataset --revision "$rev" --local-dir "$dl" \
    2>&1 | tee "$out/download_lerobot.log"
fi
if [[ ! -f "$dl/$mimic_src" ]]; then
  "$hf" download "$repo" "$mimic_src" --repo-type dataset --revision "$rev" --local-dir "$dl" \
    2>&1 | tee "$out/download_mimic.log"
fi
ln -sfn "$dl/$prefix" "$lerobot_link"
ln -sfn "$dl/$mimic_src" "$mimic_link"

cp "$dl/$prefix/meta/tasks.jsonl" "$out/tasks.jsonl"
echo "tasks.jsonl:"
cat "$out/tasks.jsonl"
tasks_json="$("$(vla_py)" -c 'import json,sys; print(json.dumps([json.loads(l)["task"] for l in open(sys.argv[1]) if l.strip()]))' "$out/tasks.jsonl")"

finish_measurement --set "tasks=$tasks_json" --set "lerobot_prefix=$prefix" --set "lerobot_dir=$lerobot_link" \
  --set "mimic_source=$mimic_link" --set "revision=$rev"
