#!/usr/bin/env bash
# Day one, V-client-tests. Do Arena's GR00T client tests import and pass without Isaac Sim? Runs them with
# two interpreters that have no Isaac Sim: the vla-isaac venv and GR00T's venv, with the Arena checkout on
# PYTHONPATH. Needs no GPU and no server: also safe on an L4 or a laptop with the venvs.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

require_env ARENA_ROOT
require_vla
begin_measurement

tests=(isaaclab_arena_gr00t/tests/test_gr00t_remote_closedloop_policy.py
       isaaclab_arena_gr00t/tests/test_gr00t_remote_closedloop_policy_runner.py)
for t in "${tests[@]}"; do require_file "$ARENA_ROOT/$t"; done

for venv in vla gr00t; do
  if [[ "$venv" == gr00t ]]; then py="$(gr00t_py)"; else py="$(vla_py)"; fi
  if [[ ! -x "$py" ]]; then
    echo "$venv venv missing ($py); skipped"
    continue
  fi
  rc=0
  (cd "$ARENA_ROOT" && PYTHONPATH="$ARENA_ROOT" "$py" -m pytest -q -p no:cacheprovider "${tests[@]}") \
    > "$out/pytest_$venv.log" 2>&1 || rc=$?
  tool client-tests --log "$out/pytest_$venv.log" --exit-code "$rc" --json-out "$out/client_tests_$venv.json"
  echo "$venv venv: exit $rc; $(tail -1 "$out/pytest_$venv.log")"
done
args=()
for venv in vla gr00t; do
  if [[ -f "$out/client_tests_$venv.json" ]]; then args+=(--json "$venv=$out/client_tests_$venv.json"); fi
done
((${#args[@]} > 0)) || die "no interpreter ran the tests"
finish_measurement "${args[@]}"
