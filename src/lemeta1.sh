#!/bin/bash
env0="CartPole-v0"
mx=20
vepochs=64
innerepochs=16
name="c/meta-vepoch"${vepochs}"-inepochs"${innerepochs}
N=4
for ((s=0;s<$N;s+=1));do
    for ((i=$s;i<$mx;i+=4));do
        echo ${i}${name}
        python3 -u pa_meta.py --outdir $name --env $env0 --gpu $(($i%2)) --seed $i --v_learn_epochs $vepochs --innerepochs $innerepochs;
    done&
done;