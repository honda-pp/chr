#!/bin/bash

env="CartPole-v0"

for ((s=0;s<$N;s+=1));do
    for ((i=$s;i<$mx;i+=4));do
        python3 -u pa.py --outdir Cperf/a2c --env $env --gpu $(($i%2)) --seed $i  --steps 20000 --eval-interval 100 > cpa.out; 
    done&
done;
env="Acrobot-v1"
for ((s=0;s<$N;s+=1));do
    for ((i=$s;i<$mx;i+=4));do
        python3 -u pa.py --outdir Aperf/a2c --env $env --gpu $(($i%2)) --seed $i  --steps 20000 --eval-interval 100 > cpa.out; 
    done&
done;