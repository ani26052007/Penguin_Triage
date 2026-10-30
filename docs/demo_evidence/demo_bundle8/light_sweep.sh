#!/usr/bin/env bash
cd ~/comp_and_research/Penguin_Triage; source ~/vla-env.sh
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
PWC=$PWD/policy-wrapper; p=5900
for s in 1 2; do
  for I in 10 30 1000; do
    p=$((p+1)); export EXTRA="light.intensity.enabled=true light.intensity.sampler_cfg.low=[$I.0] light.intensity.sampler_cfg.high=[$I.0]"
    rm -rf ~/Penguin_Triage_demo/par_light${I}_s$s
    ~/record_cell3.sh $PWC par_light${I}_s$s 10 $s $p 0 &
    sleep 60
  done
  wait
done
echo LIGHTDONE
