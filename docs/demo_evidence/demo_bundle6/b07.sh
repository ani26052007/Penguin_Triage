#!/usr/bin/env bash
until grep -q ALLDONE ~/b_suite.log; do sleep 30; done
cd ~/comp_and_research/Penguin_Triage; source ~/vla-env.sh
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
~/record_cell2.sh /tmp/pw-bug07 par_bug07_s1 10 1 5801 0 & sleep 60
~/record_cell2.sh /tmp/pw-bug07 par_bug07_s2 10 2 5802 0 & wait
echo B07DONE
