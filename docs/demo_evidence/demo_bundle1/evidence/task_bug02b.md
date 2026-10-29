Incident report (generated from measured runs, router-style summary)

System: GR00T N1.6 policy served over ZMQ, driven by the policy_wrapper client in this repo (put the ranch bottle in the fridge and close the door).
Symptom: with the same checkpoint, instruction and seeds, Arena's stock client succeeds 3/4 episodes; this repo's client succeeds 0/4.
Router evidence: infra and config checks are clean. Observation contract check: the images sent to the server are orientation-inconsistent with the simulator camera frames (the image the policy sees is rotated/flipped relative to the render).
Localization hint: the observation dict sent to the server is assembled in policy_wrapper/policy/gr00t_core.py, function build_gr00t_policy_observations. Compare how each camera frame is placed into policy_observations["video"] with the raw frame.
Task: fix the bug with the smallest possible change (a few lines at most). Do not change config values, tests, the server, or other files. Run `python -m compileall -q policy_wrapper` before you finish.
