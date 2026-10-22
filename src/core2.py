from pathlib import Path
import core1
import json
def read_numbers(path):
    lis = []
    with open(path) as f:
        for line in f:
            cleaned = line.strip()
            if not cleaned:
                continue
            
            x = float(cleaned)
            lis.append(x)
    return lis
    

def safe_mean(xs):
    if not xs:
        return None
    else:
        avg = core1.mean(xs)
    return avg

def write_summary(path, xs):
    with open(path, "a") as f:
        f.write(f"n={len(xs)} mean={safe_mean(xs)} \n")

def load_config(path):
    with open(path) as f:
        x = json.load(f)
    return x



