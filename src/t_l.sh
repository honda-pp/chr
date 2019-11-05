#!/bin/bash
env="CartPole-v0"
mx=1000
innerepochs=4
N=8
for ((s=1;s<=$N;s+=1));do
    for ((i=$s;i<$mx;i+=4));do
        echo ${i}${name}
        python3 -u meta_opt.py --env $env --gpu $(($i%2)) --t $[i*s] --innerepochs $innerepochs;
    done&
done;