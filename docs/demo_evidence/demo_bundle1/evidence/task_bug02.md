Incident report (generated from measured runs, router-style summary)

System: GR00T N1.6 policy served over ZMQ, driven by the policy_wrapper client in this repo (put the ranch bottle in the fridge and close the door).
Symptom: with the same checkpoint, instruction and seeds, Arena's stock client succeeds 3/4 episodes; this repo's client succeeds 0/4. The robot moves but does not complete the task.
Router evidence: infra and config checks are clean (same server, checkpoint, instruction, replan and denoising settings). The failure is in the observation path: the images the policy receives differ from what the simulator camera renders.
Task: find the bug in the client's observation preprocessing in this repo and fix it with the smallest change. Do not change config values, tests or the server. Run `python -m compileall -q policy_wrapper` before you finish.
