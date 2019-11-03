#!/bin/bash
env0="CartPole-v0"
env="MountainCar-v0"
mx=20
for ((i=0;i<$mx;i+=2));do
    echo ${i}"cmeta0"
    python3 -u pa_meta.py --outdir c/meta --env $env0 --gpu 0 --seed $i > cme0.out
done&
for ((i=1;i<$mx;i+=2));do
    echo ${i}"cmeta1"
    python3 -u pa_meta.py --outdir c/meta --env $env0 --gpu 1 --seed $i > cme1.out
done&
for ((i=0;i<$mx;i+=2));do
    echo ${i}"mmeta0"
    python3 -u pa_meta.py --outdir m/meta --env $env --gpu 0 --seed $i> mme0.out
done&
for ((i=1;i<$mx;i+=2));do
    echo ${i}"mmeta1"
    python3 -u pa_meta.py --outdir m/meta --env $env --gpu 1 --seed $i> mme1.out
done;