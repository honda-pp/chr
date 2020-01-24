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
            echo $l |xargs python3 -u off_and_perf.py --md_outdir "trained" --base_path "dt" --steps 20000 --eval-interval 100 --env $env --seed $i --gpu $(($i%2))
        done;
    done&
done;