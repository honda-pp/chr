#!/bin/bash
env0="CartPole-v0"
mx=2000
name="data"
N=8
for ((s=0;s<$N;s+=1));do
    for ((i=$s;i<$mx;i+=$N));do
        echo ${i}${name}
        python3 -u make_data.py --outdir $name --gpu $(($i%2)) --seed 0 --load $i;
    done&
done;