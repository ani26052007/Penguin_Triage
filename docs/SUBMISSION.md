# Submission text

Ready-to-paste material for the hackathon form. Items marked TODO need the author's real input. The measured numbers below come from one GPU session on 2026-10-29; the status table in the root `README.md` is the source of truth.

## Project description (about 300 words)

Penguin Triage answers a question every robot-learning team asks after a failed simulation run: which layer is at fault? A vision-language-action (VLA) policy fails a task in Isaac Sim. The cause may be a bug in the policy-side code or configuration (a swapped colour channel, a wrong action chunk length, a stale normalisation file), a gap in the training data (the policy never saw this lighting), or something that should not be "fixed" at all (a simulator fault, a missing skill). Choosing wrongly wastes the downstream spend, for example a fine-tuning run to fix a one-line bug.

The system has three parts. The router (`vla-router`) runs deterministic contract checks (observation, action, configuration, staleness) and counterfactual simulator replays, and combines them with a fixed decision table. No model decides the route. For code and configuration bugs, hagent (`hybrid-agent`), a bash-only repair agent, takes the verdict and the evidence and patches the code. It uses two Nemotron models hosted on Nebius Token Factory: `nvidia/Nemotron-3_5-Lightning` drives almost every step, and `nvidia/Nemotron-3-Ultra-550b-a55b` is consulted rarely, for planning, stuck advice and final review, under hard step, cost and big-call budgets. A verification ladder then checks the patch: Rung 0 without the simulator (diff, lint, tests, smoke run), Rung 1 with re-run episodes, Rung 2 by re-routing the fixed system.

The simulation side (`vla-isaac`) provides an Isaac Lab Arena backend for the policy GR00T N1.6, an Isaac Lab Mimic reference controller, experiment tools, a data pipeline, a GPU runbook, and a ledger of 14 injected bugs and 2 combined cases with expected verdicts, which serve as the evaluation set. `policy-wrapper` is the repair target, derived from Arena's Apache-2.0 GR00T client without copying any GR00T code.

Results from one real GPU session on 2026-10-29 (RTX PRO 6000 Blackwell, 97 GB), at small scale. With GR00T N1.6 behind our policy-wrapper client (10 episodes per seed, one environment per process), the clean run succeeded in 32 of 40 episodes (seeds 1-4). With BUG-02 injected (the policy image rotated 180 degrees) it succeeded in 0 of 30. After hagent removed the faulty line it succeeded in 28 of 40; the repository was byte-identical to the clean one, and 0.70 against 0.80 is within noise (Fisher p about 0.4), so the result is a return to the clean range, not a proven exact recovery. The router's observation check, run on real frames from 3 episodes per condition, flagged BUG-02 (180-degree rotation, strength 0.99) and the RGB/BGR swap BUG-01 (strength 1.00) and stayed silent on a held-out clean run. BUG-01 is a finding in itself: the router flags it, but the policy tolerates the swap (22 of 30 successes), so it is not a failure to repair.

hagent used Nemotron-3_5-Lightning and Nemotron-3-Ultra-550b-a55b on Nebius Token Factory: 22 small-model and 3 big-model calls, about $0.014 in total. A first attempt with a vaguer task edited the wrong file (65 small-model calls, about $0.07 in total). On a separate SWE-rebench-V2 benchmark (23 tasks) the hybrid mode did not beat the small model alone (13/39 against 14/39 on the 13-task fair set).

Limits, stated plainly: the hand-off from router to hagent was manual. The repair task was written by hand from measured results and contained a localization hint (the function and file), so the system was not shown locating the bug. The router check ran from a script outside the backend, on 3 episodes per condition, one task. The decision table, replays, Rung 1 and 2 and the incident loop were not run on real data. The pilot gate has not been established, the Isaac Lab Mimic reference fails at camera init on both cpu and cuda:0, and Docker was not used on the GPU box. Software tests (814, 175, 643 and 40) run against mocks and fakes. Evidence files are listed in the README; the README states what is and is not verified.

## Feedback: Nebius Token Factory

TODO(author): write from your own experience. Suggested prompts, not content:

- Setup: how long from account to first successful `Nemotron-3_5-Lightning` call; anything unclear in the docs or the OpenAI-compatible endpoint.
- Model behaviour: whether `chat_template_kwargs` / `enable_thinking` worked as expected; replies cut off at `max_tokens` (the hagent commit log mentions raising it); tool or format problems.
- Latency, rate limits and errors seen, with numbers if you recorded any (the repo has none).
- Pricing and cost tracking: whether the `usage` block was complete and the configured prices matched the bill.
- Anything you wanted that was missing.

Facts the repository can support, for you to build on (not opinions): Token Factory served `Nemotron-3_5-Lightning` and `Nemotron-3-Ultra-550b-a55b` successfully on the first try in the 2026-10-29 hagent runs (the trajectories show completed calls with `usage` for both models). Cost per fix was about $0.014 (attempt 2; small $0.0072, big $0.0064) and about $0.07 for the failed first attempt. In the SWE-rebench-V2 benchmark the big model added nothing measurable in hybrid mode.

## Feedback: Nebius AI Cloud

TODO(author): only if you used it. The repo does not say which provider the GPU box was; the runbook names none. Delete this section if not used.

## Feedback: NVIDIA tools (Nemotron, GR00T N1.6, Isaac Lab Arena, Isaac Sim, Isaac Lab Mimic)

TODO(author): write from your own experience. Points the repo's docs record as open, to confirm or discard:

- Whether the Nemotron model ids and the Nemotron-based router jobs work as planned (`vla-router/router_models.example.toml` has placeholders; `docs/plan.md` lists the unverified endpoints).
- Blackwell (sm_120): the shipped GR00T torch 2.7.1+cu126 had no sm_120 kernels; `torch==2.7.1+cu128` from the PyTorch cu128 index fixed it. Say whether this is documented anywhere you looked.
- Isaac Lab Arena: whether the documented 0.98 success over 50 episodes reproduces, and which command produced it (`docs/PILOT_AND_ORDER.md` says the page does not say).
- The Vulkan prerequisite (`libnvidia-gl-<driver>`, `VK_ICD_FILENAMES`) for Isaac Sim on your driver. The session hit it, the Blackwell torch issue above, and an HDF5 file-lock collision when three simulators started at once (h5py "unable to lock file"); these are NVIDIA-stack friction the author met, to describe in your own words.
- Mimic: the override check failed at camera init on both `--device cpu` and `--device cuda:0` (`Invalid object in Py_Graph in getWrappedGraphFromNode`).
- Mimic (continued): whether the lighting override takes effect and the seed flag exists (`V-mimic-override`, `V-mimic-seed`).

## What was updated during the submission period

TODO(author): confirm the submission-period start date; the wording below uses the git history only.

The project began as a written plan committed on 2026-10-07 (`docs/plan.md`; the initial commits are `initial plan` and `Initial commit`). The hagent repair agent was started on 2026-10-18 in its own repository. Everything else in this repository was built from 2026-10-23 onward: the router and verification ladder (cleaned-up snapshot 2026-10-23, stage fixes 2026-10-24), then the simulation side in eight phases between 2026-10-27 and 2026-10-28 (the `vla-isaac` package and backend, `policy-wrapper`, instrumentation and trace sources, the reference controller and training spec, experiment tools, the data pipeline, the GPU runbook, and the injected-bug ledger), followed by the hagent changes to its prompts, detector, budget handling and Nebius configuration on 2026-10-25 to 2026-10-27. On 2026-10-29 the first real-GPU session ran (stock Arena client, policy-wrapper, BUG-02 and BUG-01 injected, the router's observation check on real frames, BUG-02 fixed by hagent on Nebius Token Factory), and the README, this document, the Vulkan and Blackwell torch checks in the GPU scripts, and the repository hygiene changes were written.

Adjust the sentence about 2026-10-07 if any of it falls before your period's start: the plan and the first hagent commits would then be pre-existing work, and the router, `vla-isaac`, `policy-wrapper` and the bug ledger are what was built during it.
