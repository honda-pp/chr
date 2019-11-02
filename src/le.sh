#!/bin/bash
env="MountainCar-v0"
mx=20
for ((i=0;i<$mx;i+=2));do
    echo ${i}"cheet0"
    python3 -u pa_vc.py --outdir m/cheet --env $env --gpu 0 --seed $i > vc0.out
done&
for ((i=1;i<$mx;i+=2));do
    echo ${i}"cheet1"
    python3 -u pa_vc.py --outdir m/cheet --env $env --gpu 1 --seed $i > vc1.out
done&
for ((i=0;i<$mx;i+=2));do
    echo ${i}"norm0"
    python3 -u pa.py --outdir m/norm --env $env --gpu 0 --seed $i> nm0.out
    
done&
for ((i=1;i<$mx;i+=2));do
    echo ${i}"norm1"
    python3 -u pa.py --outdir m/norm --env $env --gpu 1 --seed $i> nm1.out
done;