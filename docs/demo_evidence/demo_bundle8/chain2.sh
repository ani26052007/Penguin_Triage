#!/usr/bin/env bash
cd ~/comp_and_research/Penguin_Triage; source ~/vla-env.sh
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export VLA_INSTRUMENT_FRAME_EVERY=5 VLA_INSTRUMENT_PROFILE_BUCKETS=16 VLA_INSTRUMENT_OBS_KEYS=policy.left_eef_pos,policy.right_eef_pos
mkdir -p ~/router_ev/bug09 ~/router_ev/clean3 ~/router_ev/clean4
VLA_INSTRUMENT_DIR=$HOME/router_ev/bug09 ~/record_cell2.sh /tmp/pw-bug09 ev_bug09 4 11 5921 0
VLA_INSTRUMENT_DIR=$HOME/router_ev/clean3 ~/record_cell2.sh $PWD/policy-wrapper ev_clean3 4 13 5922 0
VLA_INSTRUMENT_DIR=$HOME/router_ev/clean4 ~/record_cell2.sh $PWD/policy-wrapper ev_clean4 4 14 5923 0
echo CHAINDONE
