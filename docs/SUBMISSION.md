# Submission text

Ready-to-paste material for the hackathon form. Items marked TODO need the author's real input. The measured numbers below come from one GPU session on 2026-10-29/30; the status table in the root `README.md` is the source of truth.

## Project description (about 300 words)

Penguin Triage asks, after a VLA policy fails in simulation: is the cause a code or configuration bug, a gap in the training data, or something that should not be "fixed" (a simulator fault, a missing skill)? A deterministic router (`vla-router`) runs contract checks and counterfactual replays and applies a fixed decision table; no model decides the route. For code and configuration bugs, hagent (`hybrid-agent`) patches the repository with two Nemotron models on Nebius Token Factory: a small model drives nearly every step, a big one is consulted rarely. For vision and action-gap verdicts a deterministic advisor lists remediation options with their status; none of its data or fine-tuning paths was run.

Results come from one GPU session on 2026-10-29/30 (RTX PRO 6000 Blackwell, a Spot VM preempted several times), at small scale, with GR00T N1.6 behind our policy-wrapper client. Clean: 32/40 episodes. Of 7 injected bugs run in the simulator, 3 break the policy (BUG-02 0/30, BUG-03 0/20, BUG-07 0/20) and 4 are tolerated or mild (BUG-06, 13/20, is not distinguishable from clean at this n). The router's observation check flagged BUG-02 and BUG-01 on real frames; its config check flagged BUG-07, BUG-05 and BUG-06 on real repos and nothing on the clean one (hand-assembled spec, outside `IsaacBackend`).

Automatic repair without a hand-written hint (router evidence, a Nemotron-written report, hagent, a scripted check; five attempts per bug, scripts outside the incident loop): BUG-02 4/5 fixed, BUG-07 2/5, BUG-03 1/5 from a symptom-only task. n = 5 gives wide intervals, no selector was run (pass@3 of 1.00, 0.90 and 0.60 assumes one), the BUG-03 check is a line check, and the BUG-07 fixes were not rerun in the simulator.

Further measurements from the same session: the router's staleness probe (constant-delay half) flags BUG-09, so the router's checks flag 6 of the 7 bugs run in the simulator, 4 of them bugs the policy tolerates or only mildly suffers from (01, 05, 06, 09); BUG-03 has no detection by design. False positives on held-out clean runs: 0 of 3 (n tiny). A dome-light sweep (policy-wrapper, 10 episodes per seed, seeds 1-2) gave 0/20 at intensity 10, 0/20 at 30, 11/20 at 1000 and 17/20 at the default on the same seeds: a real vision-failure cell found with no code change (the intensity-1000 drop is suggestive, Fisher p about 0.1). The observation check also fires on the dark cell (value range and colour balance), so contract evidence alone cannot separate a vision gap from a code bug; the replay that does (one-factor restore) was not run. `decide()` on real evidence returned `undetermined` (row 10) for BUG-02, BUG-07 and BUG-03 because the reference-controller replay (G2) is missing; that is designed conservative behaviour, not a routing success. hagent was also run on a SWE-rebench-V2 subset (23 tasks): on the 13 fair tasks small solved 14/39 and hybrid 13/39, so the big model added nothing measurable there.

Not established: the incident loop, Rung 1 and 2, Branch G and Branch C, `IsaacBackend` end to end, the 50-episode pilot gate (not formally run; the smaller sample meets its two conditions). No vision failure was repaired, Isaac Lab Mimic does not start on this box, and the data pipeline and fine-tuning are implemented, not run, with no claim about their effect. The plan's headline claim (correct routing and restored held-out success at lower cost than a fixed expensive recipe) is untested. Software tests (vla-router 835, hagent 175, vla-isaac 676 and policy-wrapper 40 passed) run against mocks and fakes.

## Feedback: Nebius Token Factory

TODO(author): write from your own experience. Suggested prompts, not content:

- Setup: how long from account to first successful `Nemotron-3_5-Lightning` call; anything unclear in the docs or the OpenAI-compatible endpoint.
- Model behaviour: whether `chat_template_kwargs` / `enable_thinking` worked as expected; replies cut off at `max_tokens` (the hagent commit log mentions raising it); tool or format problems.
- Latency, rate limits and errors seen, with numbers if you recorded any (the repo has none).
- Pricing and cost tracking: whether the `usage` block was complete and the configured prices matched the bill.
- Anything you wanted that was missing.

Facts the repository can support, for you to build on (not opinions; hagent was also run many times on the SWE-rebench-V2 benchmark, laptop, Token Factory, results in the root `README.md`): Token Factory served `Nemotron-3_5-Lightning` and `Nemotron-3-Ultra-550b-a55b` successfully on the first try in the 2026-10-29 hagent runs (the trajectories show completed calls with `usage` for both models). Cost per fix was about $0.014 (attempt 2; small $0.0072, big $0.0064) and about $0.07 for the failed first attempt. In the SWE-rebench-V2 benchmark the big model added nothing measurable in hybrid mode.

## Feedback: Nebius AI Cloud

TODO(author): only if you used it. The repo does not say which provider the GPU box was; the runbook names none. Delete this section if not used.

## Feedback: NVIDIA tools (Nemotron, GR00T N1.6, Isaac Lab Arena, Isaac Sim, Isaac Lab Mimic)

TODO(author): write from your own experience. Points the repo's docs record as open, to confirm or discard:

- Whether the Nemotron model ids and the Nemotron-based router jobs work as planned (`vla-router/router_models.example.toml` has placeholders; `docs/plan.md` lists the unverified endpoints).
- Blackwell (sm_120): the shipped GR00T torch 2.7.1+cu126 had no sm_120 kernels; `torch==2.7.1+cu128` from the PyTorch cu128 index fixed it. Say whether this is documented anywhere you looked.
- Isaac Lab Arena: whether the documented 0.98 success over 50 episodes reproduces, and which command produced it (`docs/PILOT_AND_ORDER.md` says the page does not say).
- The Vulkan prerequisite (`libnvidia-gl-<driver>`, `VK_ICD_FILENAMES`) for Isaac Sim on your driver. The session hit it, the Blackwell torch issue above, and an HDF5 file-lock collision when three simulators started at once (h5py "unable to lock file"); these are NVIDIA-stack friction the author met, to describe in your own words.
- Mimic: the override check failed at camera init on both `--device cpu` and `--device cuda:0` (`Invalid object in Py_Graph in getWrappedGraphFromNode`); a nominal run (no lighting override, no `--viz none`) fails the same way, and app-profile greps showed no difference between the Mimic and `policy_runner` launchers.
- Mimic (continued): whether the lighting override takes effect and the seed flag exists (`V-mimic-override`, `V-mimic-seed`).

## What was updated during the submission period

TODO(author): confirm the submission-period start date; the wording below uses the git history only.

The project began as a written plan committed on 2026-10-07 (`docs/plan.md`; the initial commits are `initial plan` and `Initial commit`). The hagent repair agent was started on 2026-10-18 in its own repository. Everything else in this repository was built from 2026-10-23 onward: the router and verification ladder (cleaned-up snapshot 2026-10-23, stage fixes 2026-10-24), then the simulation side in eight phases between 2026-10-27 and 2026-10-28 (the `vla-isaac` package and backend, `policy-wrapper`, instrumentation and trace sources, the reference controller and training spec, experiment tools, the data pipeline, the GPU runbook, and the injected-bug ledger), followed by the hagent changes to its prompts, detector, budget handling and Nebius configuration on 2026-10-25 to 2026-10-27. On 2026-10-29 the first real-GPU session ran (stock Arena client, policy-wrapper, BUG-02 and BUG-01 injected, the router's observation check on real frames, BUG-02 fixed by hagent on Nebius Token Factory), and the README, this document, the Vulkan and Blackwell torch checks in the GPU scripts, and the repository hygiene changes were written. On 2026-10-29/30 the session continued with a bug suite (BUG-03, 05, 06, 07, 09), the router's config check on real repos and automatic-repair runs for BUG-02, BUG-07 and BUG-03; the remediation advisor and the pass@k, config-check and auto-repair scripts were added afterwards. Later on 2026-10-30 the session added the lighting sweep, the dark-cell and false-positive checks, the staleness probe on real runs and `decide()` on real evidence; `policy-wrapper` became a git submodule.

Adjust the sentence about 2026-10-07 if any of it falls before your period's start: the plan and the first hagent commits would then be pre-existing work, and the router, `vla-isaac`, `policy-wrapper` and the bug ledger are what was built during it.

## Plan versus what was done (short version)

The full table, with a reason per item, is in the root `README.md` ("Plan versus what was done").

- **Run on real data:** the policy under test (clean 32/40); 7 injected bugs in the simulator (3 break the policy: 02, 03, 07); the observation check (01, 02, dark cell), config check (05, 06, 07) and constant-delay staleness probe (09); 0 of 3 false positives; automatic repair (BUG-02, BUG-07, BUG-03; n = 5); the lighting sweep; `decide()` on real evidence (`undetermined`, G2 missing); the hagent benchmark on SWE-rebench-V2.
- **Tested on fakes only:** the incident loop, Rung 0/1/2, `IsaacBackend` end to end, the experiment tools, the data pipeline, mixer, held-out split, launcher, `finetune_dryrun`, the remediation advisor, the other 7 ledger bugs and the 2 combined cases.
- **Not done:** Stage 0 and Stage 2 driven by the router on the simulator; Branch G and Branch C; reference-controller qualification (Mimic is broken); clean-vs-incident mode gap and replay agreement; the variable-delay staleness half, the sanity check and joint-tracking on real data; calibration, the router evaluation set and confusion matrix, held-out bug types and ablations; Rung 0/1/2 on real runs; the manual container leak check (no Docker on the VM); the vision-fix half (data ladder, fine-tune, held-out evaluation, cost baseline, Cosmos, LoRA, factor search, hunk bisect); the DROID fallback decision; the 50-episode pilot gate; wrapper parity at a meaningful size.
- **Out of scope by design (plan section 2):** sim-to-real, a new sensor modality, action-gap repair beyond the advisor menu.

The narrower measured claim: on a real GR00T N1.6 policy, injected code bugs change success in some cases (3 of 7 break it); the router's contract and delay checks flag 6 of 7 on real data, including 4 that barely change success, with 0 of 3 false positives on held-out clean runs; the same observation check also fires on a pure lighting shift, so contract evidence alone cannot route; a Nemotron-based agent on Nebius Token Factory repairs located bugs more often than an unlocated one (4/5, 2/5 vs 1/5); and lighting alone can drive success from 0.85 to 0. Routing between categories, verification rungs and retraining were designed and tested on fakes but not demonstrated.
