#!/bin/bash
IFS="
"
list=(`cat args.txt`)
env="CartPole-v0"
mx=20
N=4
for ((s=0;s<$N;s+=1));do
    for ((i=$s;i<$mx;i+=4));do
        for l in "${list[@]}";do
            echo ${i}-$l
            echo $l |xargs python3 -u off_and_perf.py --outdir Cperf --md_outdir "Ctrained" --base_path "cdt" --env $env --seed $i --gpu 1 #$(($i%2))
        done;
    done&
done