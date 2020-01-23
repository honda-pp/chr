#!/bin/bash
#env="CartPole-v0"
env="Acrobot-v1"
name="a/a2c"
mx=20
N=4
for ((s=0;s<$N;s+=1));do
    for ((i=$s;i<$mx;i+=4));do
        echo ${i}${name}
        python3 -u pa.py --outdir $name --env $env --gpu $(($i%2)) --seed $i;
    done&
done;