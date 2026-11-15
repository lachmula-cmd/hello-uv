from pathlib import Path
import urllib.request


BASE = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"
FILES = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz",  "t10k-labels-idx1-ubyte.gz"]

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

for name in FILES:
    target = data_dir / name
    if target.exists():
        print("already have")
        continue
    print("downloading:", name)
    urllib.request.urlretrieve(BASE + name, target)
    print("saved", target, target.stat().st_size, "bytes")