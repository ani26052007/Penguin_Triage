#!/usr/bin/env bash
cd ~/comp_and_research/Penguin_Triage; source ~/vla-env.sh
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
STATS=$(find ~/data/models/pilot -name statistics.json | head -1); echo "stats: $STATS"
for b in 03 05 06 07 09; do
  rm -rf /tmp/pw-bug$b; extra=""; [[ $b == 07 ]] && extra="--checkpoint-stats $STATS"
  env -u PYTHONPATH $VLA_VENV/bin/python vla-isaac/bugs/inject.py inject --base policy-wrapper --out /tmp/pw-bug$b --bug BUG-$b $extra 2>&1 | grep -E '"audit"|rror'
done
i=0
for b in 03 05 06 07 09; do for s in 1 2; do
  i=$((i+1)); rm -rf ~/Penguin_Triage_demo/par_bug${b}_s$s
  ~/record_cell2.sh /tmp/pw-bug$b par_bug${b}_s$s 10 $s $((5700+i)) 0 &
  sleep 60
  while [[ $(jobs -rp | wc -l) -ge 3 ]]; do sleep 10; done
done; done
wait; echo ALLDONE
