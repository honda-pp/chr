import json, os, re, glob, shutil
import numpy as np
import matplotlib.pyplot as plt

def seed_used(seed, path):
    for dr in glob.glob(path):
        for file in glob.glob(dr+"/*/"):
            if os.path.exists(file+"args.txt"):
                with open(file+"args.txt", 'r') as f:
                    if seed == json.loads(f.read())["seed"]:
                        return os.path.exists(file+"best")

def comp(means, stds):
    mean = means.mean(0)
    N = 14
    std = np.sqrt(((N - 1) * (np.power(stds, 2).sum(0)) + N * np.power(means - mean, 2).sum(0)) / (N * stds.shape[0] - 1))
    return mean, std

def comp_perf(folders):
    for folder in folders:
        files = glob.glob(folder+"/*/sco*")
        num = len(files)
        means = np.zeros([num, 200])
        stds = np.zeros([num, 200])
        for j, file in enumerate(files):
            with open(file, 'r') as f:
                f.readline()
                lines = f.readlines()
                length = len(lines)
            for i, line in enumerate(lines):
                means[j,i], stds[j,i] = map(float, line.split("\t")[3:6:2])

        mean, std = comp(means, stds)
        label = folder[2:]
        """
        label = re.sub("norm", "A2C", label)
        label = re.sub("cheet", "reference", label)
        label = re.sub("meta.*", "A2C (meta)", label)
        """
        plt.plot(range(200), mean, label=label)
        plt.fill_between(range(200), mean-std, mean+std, alpha=0.3)
    plt.ylabel("reward")
    plt.xlabel("steps (×10^3)")
    plt.legend(loc='lower right')

def del_missed_dir(folders):
    for fol in folders:
        fls = glob.glob(fol+"/*")
        for fs in fls:
            if not os.path.exists(fs+"/best"):   
                print(fs)         
                shutil.rmtree(fs)
        try:
            os.rmdir(fol)
            print(fol)
        except OSError:
            pass

def del_overlap(folders):
    for path in folders:
        for seed in range(20):
            i = 0
            for file in glob.glob(path+"/*/a*"):
                with open(file,'r') as f:    
                    if json.load(f)["seed"] == seed:
                        i += 1
                        if i >= 2:
                            print(file, seed, i)
                            shutil.rmtree(re.sub("/args.txt", "", file))
