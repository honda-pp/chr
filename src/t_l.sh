#!/bin/bash
env="CartPole-v0"
mx=1000
innerepochs=4
t_v_learn_epochs=200
N=4
for ((s=0;s<$N;s+=1));do
    for ((i=$s;i<$mx;i+=$N));do
        echo ${i}${name}
        python3 -u meta_opt.py --env $env --gpu $(($i%2)) --t $i --innerepochs $innerepochs --t_v_learn_epochs $t_v_learn_epochs > "log/out"${i}".txt";
    done&
done;