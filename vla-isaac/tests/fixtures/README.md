# Test fixtures (FIXTURE data, not from the real checkpoint or dataset)

- `checkpoint_gr1/`: files shaped like a GR00T N1.6 checkpoint (`config.json`, `processor_config.json`,
  `statistics.json`, `embodiment_id.json`). Every value is made up or a GR00T default; each JSON
  file except `embodiment_id.json` carries a `_comment` saying so. `processor_config.json`'s modality
  config for `gr1` is Arena's `gr1_arms_only_data_config.py` read with
  `vla_isaac.specs.arena_data_config`, so the nominal fixture agrees with Arena.
  The real files are checked by V-ckpt-cfg and V-stats-format.
- `dataset_gr1/meta/`: a LeRobot `meta/info.json` and `meta/tasks.jsonl`. The task string is the
  eval instruction ("sauce bottle"), chosen so the nominal spec has no instruction violation; the
  real string is V-train-instr (Arena's converter YAML says "ranch dressing bottle").
- `list_variations_gr1_fridge.txt`, `fake_rep_server.py`: see their headers.
- Mimic logs are written by the tests (`tests/test_mimic.py`); their progress lines follow the
  PLACEHOLDER pattern of V-mimic-out.
