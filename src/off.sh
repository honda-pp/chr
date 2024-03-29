#!/bin/bash
env="CartPole-v0"
mx=20
N=1
ve_list=(1 4 8 16)
is_list=(0.01 0.05 0.1 0.5)
ie_list=(1 4 8 16)
os_list=(0.01 0.05 0.1 0.5)
count=1
for ve in "${ve_list[@]}";do
    v_learn_epochs=$ve
    for ie in "${ie_list[@]}";do
        innerepochs=$ie
        for os in "${os_list[@]}";do
            outerstepsize=$os
            for is in "${is_list[@]}";do
                innerstepsize=$is
                for ((s=0;s<$N;s+=1));do
                    for ((i=$s;i<$mx;i+=N));do
                        name="c/off-ve"${v_learn_epochs}"-ie"${innerepochs}"-os"${outerstepsize}"-is"${innerstepsize}
                        echo ${i}${name}                        
                        python3 -u off_and_perf.py --outdir $name --env $env --gpu $(($count%2)) --seed $i --v_learn_epochs $v_learn_epochs --innerepochs $innerepochs --innerstepsize $innerstepsize --outerstepsize $outerstepsize >> "log/${name}.out";
                    done;
                done;
            done;
        done;
    done&
    count=$((++count))
done;