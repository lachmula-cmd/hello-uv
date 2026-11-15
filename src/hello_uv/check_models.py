import numpy as np
import hello_uv.models as models

results = []


def report(label, ok, detail=""):
    print(("PASS     " if ok else "FAIL     ") + label + (f"  ({detail})" if detail and not ok else ""))
    results.append(ok)


def get(name):
    obj = getattr(models, name, None)
    if obj is None:
        print(f"MISSING  {name} is not defined in models.py")
        results.append(False)
    return obj


X = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 0.0, 1.0], [7.0, 2.0, 3.0]])  # 4 cells, 3 genes

Base = get("EmbeddingModel")
if Base:
    try:
        m = Base("base")
        report("1 base class stores its name", m.name == "base", f"got {m.name!r}")
        try:
            m.embed(X)
            report("1 base embed() refuses to run", False, "no error was raised")
        except NotImplementedError:
            report("1 base embed() refuses to run", True)
    except Exception as e:
        report("1 base class", False, f"{type(e).__name__}: {e}")

Ident = get("IdentityModel")
if Ident and Base:
    try:
        m = Ident()
        report("2 IdentityModel is an EmbeddingModel", isinstance(m, Base))
        report("2 IdentityModel name is 'identity'", m.name == "identity", f"got {m.name!r}")
        report("2 IdentityModel returns X unchanged", np.array_equal(m.embed(X), X))
    except Exception as e:
        report("2 IdentityModel", False, f"{type(e).__name__}: {e}")

Centre = get("MeanCentreModel")
if Centre and Base:
    try:
        m = Centre()
        out = m.embed(X)
        report("3 MeanCentreModel is an EmbeddingModel", isinstance(m, Base))
        report("3 MeanCentreModel name is 'mean-centre'", m.name == "mean-centre", f"got {m.name!r}")
        report("3 every column now has mean 0", out.shape == X.shape and np.allclose(out.mean(axis=0), 0), f"column means {np.round(out.mean(axis=0), 3)}")
    except Exception as e:
        report("3 MeanCentreModel", False, f"{type(e).__name__}: {e}")

Proj = get("RandomProjectionModel")
if Proj and Base:
    try:
        a, b, c = Proj(2, seed=0), Proj(2, seed=0), Proj(2, seed=1)
        report("4 RandomProjectionModel name is 'random-2'", a.name == "random-2", f"got {a.name!r}")
        out = a.embed(X)
        report("4 output shape is (cells, dim) = (4, 2)", np.shape(out) == (4, 2), f"got {np.shape(out)}")
        report("4 same seed gives the same output", np.allclose(out, b.embed(X)))
        report("4 a different seed gives a different output", not np.allclose(out, c.embed(X)))
    except Exception as e:
        report("4 RandomProjectionModel", False, f"{type(e).__name__}: {e}")

embed_all = get("embed_all")
if embed_all and Ident and Centre and Proj:
    try:
        got = embed_all([Ident(), Centre(), Proj(2, seed=0)], X)
        want = {"identity": (4, 3), "mean-centre": (4, 3), "random-2": (4, 2)}
        report("5 embed_all gives each model's output shape", got == want, f"got {got}")
    except Exception as e:
        report("5 embed_all", False, f"{type(e).__name__}: {e}")

print(f"\n{sum(results)} of {len(results)} checks passed")