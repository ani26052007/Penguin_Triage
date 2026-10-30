# Future plans and improvements

This is what I plan to work on after my end-semester exams. It is my list of what is still open, in the order I intend to tackle it, updated to the state after the GPU session of 2026-10-29/30. What has been measured is in [TECHNICAL_DETAILS.md](TECHNICAL_DETAILS.md); the design choices behind the current code are in [DECISIONS.md](DECISIONS.md).

Where it stands today: the router's contract checks ran on real GR00T N1.6 data and flagged 6 of 7 injected bugs, and hagent on Nebius Token Factory repaired three bugs automatically (4/5, 2/5, 1/5). But `decide()` returns `undetermined` on real evidence because the reference-controller replay is missing, no vision failure has been repaired, and the incident loop has only run on mocks. The plan's headline claim is still untested, and closing that gap is the point of this list.

## 1. Close the routing loop on real hardware (first)

1. **Get a reference controller (G2).** Without one, the decision table cannot rule out a simulator or asset fault and stays `undetermined`, which is the designed behaviour.
   - Isaac Lab Mimic fails at camera init on my Blackwell box (`Invalid object in Py_Graph`) with both `--device cpu` and `cuda:0`. My one untested lead: Mimic uses `gr1_pink`, while `policy_runner` (which creates cameras fine) uses `gr1_joint`.
   - Cheaper stand-in: use the stock Arena client (3/4 on this scene) as a "stock-client reference" for G2.
   - Either way, the reference needs a qualification step. Nothing produces `reference_unqualified` yet, and a scripted controller may not fit the policy's nominal band.
2. **Run the one-factor-restore replay on the dark-lighting cell.** The observation check fires on the dark cell exactly as it would on a preprocessing bug. The replay is the part of the design that separates the two, and I have only done it by hand from logged runs.
3. **Run `IsaacBackend` end to end**, then drive `route()` and the incident loop on the simulator: Stage 0 and Stage 2, Branches G and C, and Rung 0/1/2. My GPU-session checks ran as scripts outside `IsaacBackend`.
4. **Use Rung 1 as a selector over repair attempts.** pass@3 (1.00 / 0.90 / 0.60) only means something with a selector; the observation and config checks on the patched repo are natural candidates.
5. **Make `run_rung1` usable as an agent tool.** It shares hagent's 60 s command timeout (too short for a real Isaac run), its runs count toward no simulator budget, the reviewer never sees its output, and no production code registers it.

## 2. Calibrate before trusting any threshold

- `fire_z = 4.0`, the metric floors, `min_strength = 0`, the behaviour thresholds and the 0.15 / (0.70, 0.95) anchors are all placeholders. I want to run the router many times on no-fault incidents, set thresholds so the false-verdict rate stays under a target, and report that rate. On the mock, sampling noise already turned 2 of 10 `action_gap` repeats into `vision_factor`.
- Nominal statistics have to be per scene condition, which means no-fault runs for every shifted condition. That costs a lot and I have not budgeted it.
- Measure replay agreement (needed for paired tests). The tool exists but has never run, and how a seed maps to the policy's sampling seed is still open.
- Use per-incident anchors instead of fixed global ones, and account for the failure anchor's own uncertainty.
- Rung 1 checks absolute levels, not improvement: a no-op patch on a partly broken target comes out `ambiguous` and spends the Rung 2 call.
- The second-cause flag is unreliable when the nominal reads `no_effect` at 10 seeds (missing in 32/50 mock runs). Branch-keyed Rung 1 rescues the loop, but the flag itself cannot be trusted yet.
- Run the 50-episode pilot gate and wrapper parity at a meaningful size. My clean 32/40 and dark-cell 0/20 meet the gate's two conditions only in a small sample.

## 3. More real data

- Run the other 7 ledger bugs and the 2 combined cases in the simulator, so far 7 of 14 bugs. BUG-10 needs its helper script fixed first: it passes `--language_instruction`, which overrides `eval.yaml`.
- Rerun BUG-07 through the fixed `inject.py` (multi-embodiment `statistics.json`). The current numbers come from a gr1-only copy.
- Rerun the BUG-07 and BUG-03 fixes in the simulator. They were scored by the config check and a line check only.
- Run the checks that have not touched real data:
  - the variable-delay half of the staleness probe (needs incident-mode runs);
  - the sanity check (its nominal builder lives in test code);
  - the action contract's joint tracking (needs tracked joint positions in the instrument files).
- Use more episodes everywhere: 3 per router condition, 20 per bug and 5 repair attempts per bug are too few for anything but direction.
- Build a router confusion matrix on real incidents (today's is sampling noise on hand-written evidence), add held-out bug types, and run the planned ablations (elimination only, plus Stage 1, VLM-only baseline).
- Vary lighting beyond intensity (colour, temperature, HDR) and test whether the 1000-intensity drop (11/20, Fisher p about 0.1) is real.

## 4. The vision half

- A short Rung A fine-tune on the dark-lighting cell, now that a real failure cell exists, followed by a held-out evaluation. The data pipeline, mixer, held-out split and launcher are implemented but have never run.
- A fixed-expensive-recipe baseline, so that "cheaper than always fine-tuning" can be measured instead of claimed.
- Rung B (Mimic regeneration) waits on item 1.1.

## 5. The repair agent (hagent)

- On my SWE-rebench-V2 subset, hybrid (13/39) did not beat small (14/39). I want to find out when the big model actually helps, or drop the claim. The reviewer is also poorly calibrated (it rated wrong patches 0.8-0.9).
- Multi-site fixes: hagent fixes the first location and stops. BUG-03 without a location (1/5) shows the same weakness.
- Run the Docker backend at least once (all container tests skip; the VM had no Docker) and do the container leak check.
- Patch robustness: use `git diff --binary`, and make patches independent of the host's git config (`diff.noprefix`).
- Leakage:
  - `RepoIndex` includes `.md` and `.json` files, so a bug ledger in the repo would become the top localization candidate;
  - `incident_id` and Rung 2's free-text reason go into the prompt unchecked;
  - `$RUNG1_CONFIG` names the original checkout and, in mock runs, spells out the fix;
  - the agent can rewrite its copy's base refs.
- The report validator misses absolute paths, extension-less paths, unbracketed ids and config-key fragments. It also accepts explanations that are real but wrong.
- Budgets:
  - add a simulator-episode cap per incident;
  - hagent's per-run caps restart every attempt, so an incident can overshoot by up to one run;
  - a crash loses the cost of the call that raised;
  - the Rung 2 and reviewer models' prices are not checked.
- Rung 0 should take `expected_files` from the router's located files.
- Decide what an unconfigured Rung 2 should do on a row-4 re-run (today the incident ends `UNRESOLVED`).

## 6. Router correctness details

- **Model outputs that can still affect routing:**
  - the config mapper's `approved` flag is never read;
  - `extra_evidence` has no provenance check;
  - the structural no-model test only sees direct absolute imports.
- **Decision table, to decide deliberately:**
  - row 1 ignores `render_physics_broken` and contract evidence when Stage 0 is restored;
  - an inconclusive G3 means "code by elimination" in Branch G but row 10 in Branch C;
  - two partial restores under a code verdict set no second-cause flag;
  - row 2 reached through evidence attaches no proving replay;
  - row-10 verdicts list evidence that `min_strength` filtered out;
  - `decide` never reads G1;
  - an undeclared scene change or a reach-phase failure ends at "no row matched";
  - noisy notes when Stage 0 stops the router.
- **Missing incident-mode logs** end at row 10. The router could replay the failing seeds in incident mode instead.
- **Config contract:**
  - a mapped key missing from the eval config is only a log warning, not an unchecked signal;
  - `True` and `1` compare equal.
- **Check naming:** the channel-order and orientation checks can name the wrong fault ("GBR" for a one-channel shift, "vertical flip" for a shuffled profile), and that wrong location goes to hagent.
- **Open-loop probe:** `PolicyOpenLoopProbe` is still a stub.
- **Cache:**
  - the key ignores which backend produced an episode;
  - scene keys can collide on `;` or `=`;
  - `1` and `1.0` give different keys;
  - `hash(CodeRef)` raises;
  - `cache_hits` counts re-reads during extension.
- **Tests:**
  - some are weaker than their names (an early-stop mutation passes; `in` instead of "only"; one report test asserts the opposite of its name);
  - the shared anchors between router and Rung 1 are a convention, not enforced;
  - untested: CLI exit code 3, the unpaired-test path and row 10 to `UNRESOLVED`;
  - the always-pass Rung 0 is still used in some loop tests.

## 7. Housekeeping

- Copy the hagent benchmark artifacts (score tables, trajectories, analysis) into `docs/hagent_benchmark/`, so the 13/39 vs 14/39 numbers can be checked from the repo.
- Run the plain-venv quick start from a fresh clone.
- Run hagent's own test suite on the GPU box (it hit missing dependencies and collection errors there).
- Settle `check_client_tests` (collection error in the vla venv, no pytest in the GR00T venv).
- Configure the optional router models (report writing, Rung 2 judging, config mapping) with verified Nemotron model ids. They are placeholders today, and routing does not depend on them.
- Small cleanups:
  - the router's `pyproject.toml` says Python >= 3.10 but needs 3.11;
  - remove a few hard-coded literals (0.5 margins, the 0.25 staleness text threshold, `n_a = 10`, truncation lengths);
  - `rung2.py` hard-codes `"C_restore:"` instead of taking it from `vocab`.

## Already done

Fixed since I first listed them:
- Stage 1 and behaviour analysis now run inside `route()`.
- Patched code is stored and rebuildable, and each attempt starts from the last accepted code.
- A patched config is read from the patched copy.
- Logs must belong to the code under test.
- Partly run checks mark their signals unchecked.
- Rung 0 is wired into the loop.
- Errors end an incident `UNRESOLVED` instead of crashing.
- Cost and big-call caps are checked per incident.
- Reviews and pass probabilities are kept.
- Rung 2 can override the route only on rows 4 and 10.
- hagent is a declared dependency.

Also closed by the GPU session: live models have now run many times (benchmark, hand-hinted BUG-02, 15 automatic repair attempts). The evidence-to-report-to-hagent path ran on real data, though as scripts outside the incident loop.
