# Project Plan: Self-Verifying Robot Repair on Nebius x NVIDIA

*Compiled on 6 Oct 2026 from the full planning discussion. Everything marked **[VERIFY]** is an assumption that must be checked before building on it.*

---

## 0. TL;DR

**One-sentence pitch:** When a VLA-controlled robot fails a task in Isaac Sim, our system works out *why* (code bug vs. data gap), fixes it the right way (patch the code, or generate data and fine-tune), and proves the fix works with thousands of parallel rollouts on Nebius, reporting the real cost.

**What makes it more than "an agent that fixes code":**
1. A **counterfactual-replay router** that uses the simulator as an oracle to decide the *kind* of failure, so the remedy is chosen on evidence.
2. A **verification ladder** that reports how often cheap checks (LLM prediction) are wrong versus real simulation, with honest numbers.
3. A **data-gap path** that closes the loop: generate targeted data, LoRA fine-tune, and show held-out improvement.
4. **Cost and scale numbers** from Nebius Serverless Jobs.

**Honest odds (my guess; I have not seen competing entries):** grand prize ~3-5% as planned; Physical AI track ~10-15% if executed cleanly. Execution and the demo video matter more than any extra feature.

---

## 1. The Event

| Item | Detail (from search results) |
|---|---|
| Name | Nebius x NVIDIA Global AI Hackathon 2026 (hosted on Devpost) |
| Submission window | 26 Aug 2026 to **30 Oct 2026, 10:00 AM PDT** |
| Judging | 1-15 Dec 2026 (asynchronous review) |
| Winners announced | around 11 Jan 2027 |
| Prizes | Grand $20,000; 2nd $10,000; 3rd $6,000 |
| Track prizes | NVIDIA Jetson Orin Nano for winners of: Coding & Agentic Engineering, Best Apps & Agents, Personal AI, **Physical AI** |
| Time left (as of 6 Oct) | about 24 days |

**Judging criteria.** Your earlier plan listed four criteria: Technological Implementation, Design, Potential Impact, Quality of the Idea. I could not find the official rubric; one aggregator said only that specifics are not provided. **[VERIFY]** on the Devpost page.

**Implication of async judging (Dec 1-15):** judges will not see you live. The **2-minute video and the README are the product.**

### What related events reward (inference, not the official rubric)

- **Nebius Serverless AI Builders Challenge** scored: technical implementation, reproducibility, educational content, product-usage depth, real-world usefulness, originality. A winner ran seven ablation jobs in parallel (20 min vs 3 h sequential), served the best model on an endpoint, and reported the actual bill ($320 including failed jobs).
- **NVIDIA Nemotron hackathons:** Nota AI won the Seoul event with a simple, clear story; Loughborough won "best use of Nemotron" with a fairly plain agentic app. Sponsor events reward *visible, meaningful* use of the sponsor stack.
- **Pattern:** a visible problem, an unmistakable demo, hard numbers (cost, speed, scale), honest reporting, and a one-sentence pitch.

---

## 2. How the Idea Evolved

### 2.1 Original idea
A robot fails. The system decides whether it is a code issue or a data issue. Code: an LLM patches it. Data: generate more data and fine-tune the policy. Then re-test.

### 2.2 First reframing (MuJoCo/robosuite version)
Headline changed to *"how cheaply can you verify a robot fix without being fooled?"* with a ladder of rungs (patch-diff check, LLM prediction, parallel rollouts, video diagnosis), bug injection for free ground truth, and an ablation of logs-only vs logs-plus-video diagnosis.

### 2.3 Honest critique that reshaped it

1. **"It's just an AI agent fixing a codebase."** Correct. Coding agents already patch bugs. The differentiator has to be the *router, verification, and data loop*, not patching.
2. **Rung 0 (compare patch to the reverse of the injected diff)** is a benchmark scoring tool, not a feature. Real users have no answer key. Keep it only for evaluation, or replace it with real fix-commits from open-source robotics repos.
3. **Rung 1 (LLM predicts the outcome from code)** is weak when a cheap simulator exists, and meaningless for neural policies. Demoted to a *patch ranker* or an *ablation*.
4. **Injected bugs are circular** (you plant, you fix, you test). Must hold out whole bug types and, ideally, test on a few real bugs.
5. **Fine-tuning the policy as a headline** is risky (noisy success rates, heavy compute). The lesson: it only counts if it is a *measured before/after on held-out conditions*.

### 2.4 Pivot to Isaac Sim + a VLA (your PDF plan)
You moved to **Isaac Sim** with a **GR00T-style VLA** and added: router, repair agent (big model plans, small model edits), a four-case rung 1, rung 2 thresholds, rung 3 video diagnosis, two data-gap categories, Dreams vs Isaac data, and gradient-routed dual LoRA.

### 2.5 Problems found in the PDF plan and how each was resolved

| # | Problem | Resolution |
|---|---|---|
| 1 | Category B was "fixed" by LoRA-compensating for latency, which contradicts "code bug means patch the code" | **Decision:** code bugs are *patched*, never trained around. LoRA is only for data gaps. |
| 2 | Router relied on `X_pred` (the model's internal object-position belief), which a VLA does not expose | **Decision:** use **counterfactual replay** in the simulator (Section 4). Probe training is out of scope. |
| 3 | Rung 1 asks an LLM to predict what a VLA will do | Use it only to **rank candidate patches** and as an **ablation** measuring prediction accuracy. Never as a gate that can skip rung 2 unchecked. |
| 4 | 60/40 vs 40/60 data split inconsistent; loss weights 1.0/1.5 | Typo. Fix in all docs; treat the mix ratio as an experiment variable. |
| 5 | Why Dreams (Cosmos) data at all when Isaac can randomize lighting? | **Decision:** run the experiment Isaac-only vs Dreams-only vs mix on held-out glossy objects, and report quality *and* GPU cost per usable episode. |
| 6 | Gradient-router hook / dual-LoRA is untested complexity | Keep as **one ablation** (single LoRA vs routed two-adapter). Default to the simple version. |
| 7 | Isaac needs RTX-class GPUs; rollouts are slow | RTX PRO 6000 is reported available. **[VERIFY]** it works for *Serverless Jobs* and that Isaac runs headless in a job container. |

### 2.6 Your stated reasoning (kept for the record)
- Using NVIDIA tools (GR00T, Isaac Sim, Cosmos) is intended to be favorable with sponsor judges. *My caveat:* judges score **effective** use; a component that an ablation shows adds nothing can hurt.
- You believe "code bug = patch, data gap = generate data and fine-tune" is fine as the routing logic. Agreed, with the Category B contradiction removed (see 2.5 #1).

---

## 3. Final Architecture

```
[Task + VLA policy in Isaac Sim]
        │  (fault injected: code/sim fault OR data gap OR none)
        ▼
[Failure detected]  →  record episode (video + logs + sim ground truth)
        ▼
[ROUTER: counterfactual replay]  (parallel Serverless Jobs)
   ├─ replay with nominal visuals ........ succeeds? → DATA/PERCEPTION GAP
   ├─ replay with injected fault removed . succeeds? → CODE/CONTROL BUG
   └─ neither succeeds ................... → AMBIGUOUS → Rung 3 (video)
        ▼
 ┌──────────────────────────┐     ┌─────────────────────────────────┐
 │ CODE PATH                │     │ DATA PATH                       │
 │ Repair agent:            │     │ Targeted data generation:       │
 │  big model = diagnose    │     │  Isaac randomization (glossy,   │
 │  small model = edit      │     │  lighting, clutter)  vs         │
 │  N candidate patches     │     │  GR00T-Dreams / Cosmos clips    │
 │  Rung 1 ranks them       │     │  vs mix                         │
 │  top-k → Rung 2          │     │ LoRA fine-tune VLA              │
 └────────────┬─────────────┘     │ → Rung 2 on held-out objects    │
              │                    └────────────────┬────────────────┘
              ▼                                     ▼
        [RUNG 2: parallel rollouts on Nebius Serverless Jobs]
        clear pass / clear fail / in-between → Rung 3
              ▼
 [Output: PR (code path) or model checkpoint + report (data path),
  with before/after clips, success rates ± std, cost]
```

---

## 4. Components in Detail

### 4.1 Environment and policy
- **Simulator:** Isaac Sim (headless in jobs; GUI/render for demo clips).
- **Policy:** a GR00T-family VLA attempting a small set of tasks. **[VERIFY]** exact model version, LoRA/fine-tune support, and inference speed.
- **Scope for the hackathon: one task (pick up a mug), one policy.** More tasks only if everything else is done.

### 4.2 Failure taxonomy (what we do and do not handle)

| Cause | Handled? | Remedy |
|---|---|---|
| Data gap / OOD (glossy mug, harsh lighting, clutter) | **Yes** | Generate data, LoRA fine-tune |
| Code or control bug (latency, gain decay, frame-transform drift, threshold) | **Yes** | Patch the code |
| Simulation environment issue | Mentioned; detected via replay only | Report, do not auto-fix |
| Sim-to-real gap | **No (explicitly out of scope)** | n/a |
| Missing manipulation skill (new object shape) | Falls under "neither replay fixes it" | Report as ambiguous; stretch goal |

### 4.3 Router: counterfactual replay
Replaces the `X_true` vs `X_pred` comparison, which assumed an internal object-position readout that VLAs do not provide.

For a failed episode (same seed, same initial state):
1. **Replay A, nominal visuals:** restore standard lighting/materials, keep the fault. Success means **perception/data gap**.
2. **Replay B, fault removed:** keep the hard visuals, remove the injected latency/gain/transform error. Success means **code/control bug**.
3. **Neither succeeds:** ambiguous, so send to Rung 3 (video diagnosis).
4. **Both succeed (unlikely):** interaction effect; report it.

Run each replay with multiple seeds (e.g., 20+) as parallel jobs so the router decision has a success rate, not a single sample. **The router's own accuracy is measurable** because you know what you injected. Report it as a confusion matrix.

> Optional later: train a probe on hidden states to regress object position. Out of scope for the core plan.

### 4.4 Code path: repair agent
- **Two roles:** a larger model plans/diagnoses (chain-of-thought), a smaller model makes the edits. Suggested: Nemotron large model for planning, Nano/Super for edits. **[VERIFY]** model availability and context limits via Nebius Token Factory.
- **Scope of editable code:** the *wrapper* around the policy (control loop, transforms, gains, timing, thresholds). The agent does not edit the neural network.
- **Candidate generation:** produce N patches (e.g., 5).
- **Rung 1 as ranker:** an LLM scores each patch's likelihood of success; top 1-2 advance to Rung 2.
- **Output:** a PR containing the patch, diagnosis, before/after clips, success rates with std, and cost.

### 4.5 Verification ladder

| Rung | What | Status in final plan |
|---|---|---|
| **0** | Compare patch to the reverse of the injected diff | **Evaluation only** (not a product feature). Optionally replaced by real fix-commits. |
| **1** | LLM predicts success from code/wrapper | **Ranker + ablation.** Never a gate that skips Rung 2. |
| **2** | Parallel rollouts in Isaac Sim | **Ground truth.** Core of the Nebius usage. |
| **3** | Cosmos Reason diagnoses video for ambiguous cases | Reserved for cases Rung 2 cannot settle. |

**Rung 1 outcomes (your four cases):**
1. True positive: goes to Rung 2 (confirms).
2. True negative: back to the repair agent.
3. False positive: Rung 2 exposes it.
4. **False negative (the "deadly" one):** a good patch gets discarded. This wastes effort but is not unsafe, because every predicted success is still validated. **Bound it empirically:** send a random 5-10% of Rung 1 rejections to Rung 2 anyway and report the measured false-negative rate.

**Rung 2 decision rule (from your plan):**
- Clear pass: mean success > 90% **and** SNR > 2.
- Clear fail: mean success < 20%.
- In-between: send to Rung 3.
- Needs enough episodes per condition for these thresholds to mean anything. **[VERIFY]** episode runtime to set the budget.

### 4.6 Rung 3: video diagnosis (Cosmos Reason)
- Input: rendered failure clip plus context.
- Output: classify as *simple code bug* (send to repair agent) or *hidden data gap* (send to data path), with a rationale.
- **Ablation:** logs-only vs logs-plus-video. This tests whether video adds information. It may not; report either way.

### 4.7 Data path

**Category A: Environmental/perception gaps** (glossy mug, harsh lighting, clutter): the arm mechanics are fine; visual-spatial alignment is not.

**Data sources (the experiment):**

| Arm | Description |
|---|---|
| Isaac-only | Domain-randomized materials/lighting in simulation, ground-truth actions |
| Dreams-only | GR00T-Dreams/Cosmos-generated clips plus inverse-dynamics pseudo-actions |
| Mix | Weighted combination (ratio is a variable, not a fixed 60/40 or 40/60) |

**Measure for each arm:** held-out-object success rate (± std over 3+ seeds), **GPU-hours and dollars per usable episode**, and number of usable episodes after filtering.

**Why this matters:** Dreams is *not* automatically cheaper. It needs video generation, likely embodiment adaptation, and inverse dynamics, and the recovered actions are noisier than simulation ground truth. Isaac randomization may win on cost *and* quality. Either result is a valid finding.

**Adversarial prompting for Dreams:** e.g., base prompt "a robot picking up a mug" vs adversarial "highly reflective glossy ceramic mug, harsh direct lighting, specular highlights, deep shadows." Generate 100+ clips, filter by success.

### 4.8 Fine-tuning
- **Method:** LoRA on the VLA. Default: a **single LoRA** on mixed data.
- **Targeting hypothesis (from your PDF):** vision-language/cross-modal attention layers for perception gaps. Treat as a hypothesis to test, not a fact. **[VERIFY]** against the GR00T architecture and tooling.
- **Ablation:** single LoRA vs gradient-routed dual adapters (System 2 attention + System 1 action head). If routing does not win, ship the simple one.
- **Success criterion:** improvement on **held-out objects/lighting not used in training**, with std and SNR reported.
- **Rule:** do not use LoRA to compensate for code bugs. Those are patched.

### 4.9 Stats and rigor
- 3+ seeds; report mean ± std and SNR.
- Treat effects with SNR < 2 as preliminary.
- Hold out entire bug types and entire object/lighting conditions.
- Report failed jobs and total cost honestly.

---

## 5. Experiments / Ablations Checklist

| # | Experiment | Why it matters |
|---|---|---|
| E1 | Router accuracy (confusion matrix) against known injected causes | Proves the router is real |
| E2 | Rung 1 prediction vs Rung 2 outcome (false-accept / false-reject rates) | Honest verification finding |
| E3 | Random 5-10% of Rung 1 rejections sent to Rung 2 | Bounds the false-negative problem |
| E4 | Isaac-only vs Dreams-only vs mix: held-out success and cost per usable episode | Justifies (or drops) Dreams |
| E5 | Single LoRA vs routed dual LoRA | Tests the gradient-router idea |
| E6 | Logs-only vs logs+video diagnosis | Tests the video claim |
| E7 | Patch success on held-out bug types (and a few real bugs if feasible) | Defends against circularity |
| E8 | Cost/scale report: rollouts run, wall-clock, dollars | Matches what Nebius judges liked |

Priority if time is short: **E1, E4, E8** (then E7, E2). Cut E5 and E6 first.

---

## 6. Timeline with Gates (6 Oct to 30 Oct)

Deadline is **30 Oct, 10:00 AM PDT**. Treat **27 Oct** as the real code freeze (convert time zones carefully; you are in IST).

| Dates | Goal | Gate (if missed, cut scope) |
|---|---|---|
| **6-8 Oct** | Verify infra: RTX 6000 on Serverless Jobs; Isaac Sim headless in container; GR00T runs inference; measure seconds per episode | If Isaac or the VLA cannot run in jobs, **switch to MuJoCo/robosuite plan immediately** |
| **9-12 Oct** | Policy attempts mug task in Isaac; **visible failure on glossy/lighting condition**; code fault injection (latency/gain) | No visible, reproducible failure by 12 Oct means simplify the task |
| **13-16 Oct** | Counterfactual-replay router working with parallel jobs; first confusion matrix | |
| **17-19 Oct** | Code path: repair agent produces patches; Rung 2 validates; first PR | Drop Rung 1 ranker if it is blocking |
| **20-23 Oct** | Data path: Isaac randomization data, then (if time) Dreams; LoRA run; held-out eval | If LoRA shows nothing by 23 Oct, report as honest negative result and move on |
| **24-26 Oct** | Cost/scale numbers; ablations that are cheap; open-source repo and README | |
| **27-29 Oct** | **Demo video, README, Devpost writeup, submission. No new features.** | |
| **30 Oct (AM PDT)** | Submit early; buffer for upload issues | |

---

## 7. Demo and Submission Plan

**Video (aim for under 3 minutes, strongest content first):**
1. 0:00-0:30: robot visibly failing (glossy mug), then the same robot succeeding after the loop. One clip each, side by side.
2. 0:30-1:15: the router deciding (replay results), and the two remedies (PR for code bug; data + LoRA for data gap).
3. 1:15-2:00: one chart with the key numbers (success before/after, ± std) and the cost/scale line ("N rollouts for $X on Nebius").
4. 2:00-2:30: honest ablation result(s), including any negative finding.
5. Close: one-sentence pitch, repo link.

**README must include:** one-command reproduction where possible, architecture diagram, results with error bars, cost table, limitations (sim-only, scope), and what is not claimed.

**What not to claim:** sim-to-real transfer; a general self-healing platform; any number without error bars.

---

## 8. Risks and Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Isaac Sim / VLA does not run in Serverless Jobs | **Critical** | Test in the first 2 days; fallback to MuJoCo plan |
| Rollouts too slow for the SNR thresholds | High | Measure early; reduce task/episode length; fewer conditions |
| Router replays inconclusive | High | Strengthen with more seeds; use Rung 3 as fallback |
| LoRA shows no held-out gain | Medium | Pre-plan the honest-negative story; keep core demo independent of it |
| Building everything at 60% | **High** | Strict scope: one task, two failure modes; cut ablations E5/E6 first |
| Circular benchmark | Medium | Hold out bug types and conditions; add a few real bugs if possible |
| Demo is only charts | High | Lead with failing/fixed clips; record clips early |
| Credit/budget limits unknown | Medium | **[VERIFY]** rules and credits now; track spend daily |
| Unchecked prior work on automated robot debugging | Medium | Do a quick literature check and cite related work honestly |
| Dreams/Cosmos pipeline heavier than expected | Medium | Isaac-only arm is the fallback; report Dreams as attempted |

---

## 9. Open Questions / Things to Verify This Week

1. Official judging rubric and rules on Devpost (credits, required tools per track, open-source requirements).
2. Does RTX PRO 6000 work for Nebius **Serverless Jobs** (not only VMs)? Cost per GPU-hour?
3. Does Isaac Sim run headless in a job container? Image size and startup time?
4. Which GR00T version, and what are its fine-tuning requirements and inference speed?
5. Seconds per episode, and therefore how many rollouts per condition fit the budget.
6. Cosmos/GR00T-Dreams: what adaptation is needed for your embodiment, and what does it cost?
7. Team size and skills (Isaac Sim, VLA training, LoRA)? This decides what to cut.
8. Which track to submit to (Physical AI is the realistic target).

---

## 10. Honest Assessment

| Criterion | Assessment |
|---|---|
| Technological Implementation | Strong potential: multiple models, real parallel compute, measured results |
| Design | The weak axis unless the demo shows robots failing and recovering in Isaac Sim; the visuals are your advantage here |
| Potential Impact | Robot debugging is a real pain, but results are sim-only; hedge claims |
| Quality of the Idea | Router-by-counterfactual-replay and measured verification are the novel parts; "agent fixes code" alone is not |

**Odds (guess):** grand prize ~3-5%, Physical AI track ~10-15%, assuming clean end-to-end execution. Lower if integration is incomplete.

**What moves the needle:**
- One tight demo (fail, fix, numbers) in the first two minutes
- Everything running end to end, even if narrow
- Real cost numbers from Nebius
- An honest negative result presented as a finding
- Open-sourced benchmark and code

**What kills it:**
- Spreading across all components at partial quality
- A circular benchmark
- A chart-only demo
- Unverified infrastructure assumptions discovered late

---

## 11. Decision Log

| Decision | Status |
|---|---|
| Use Isaac Sim + GR00T (NVIDIA stack) | Decided by you; keep only components that earn their place via ablation |
| Router decides remedy: code bug means patch, data gap means data + fine-tune | Decided |
| Code bugs are patched, not trained around | Decided (resolves PDF contradiction) |
| Router via counterfactual replay, not `X_pred` | Recommended; adopt |
| Rung 1 = patch ranker + ablation, never a gate | Recommended; adopt |
| Bound false negatives by sampling 5-10% of rejections | Recommended; adopt |
| Run Isaac vs Dreams vs mix experiment | Agreed |
| Gradient-routed dual LoRA = ablation only | Agreed on ablation; default to single LoRA |
| Sim-to-real out of scope | Decided |
| Fine-tune the policy as a core part | Decided by you; must be measured on held-out conditions with a Day-23 fallback |
| Fix the 60/40 vs 40/60 inconsistency | Writing error; fix everywhere |

## 12. Important Links 

https://nebius.com/solutions/physical-ai-and-robotics
https://nebius.com/blog/posts/run-physical-ai-workflows-not-glue-code

