#!/bin/bash
env="CartPole-v0"
mx=1000
innerepochs=16
vepochs=64
name="vmeta"
N=4
for ((s=0;s<$N;s+=1));do
    for ((i=$s;i<$mx;i+=$N));do
        python3 -u meta_opt.py --env $env --gpu $(($i%2)) --t $i --innerepochs $innerepochs --v_learn_epochs $vepochs > "log/out"${i}".txt";
    done&
done;