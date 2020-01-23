import json, os, glob

def seed_used(seed, path):
    for dr in glob.glob(path):
        for file in glob.glob(dr+"/*/"):
            with open(file+"args.txt", 'r') as f:
                if seed == json.loads(f.read())["seed"]:
                    return os.path.exists(file+"best")
