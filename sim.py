import json, math, sys
import numpy as np

AREA = 20.0          # km x km city area
N_DON = 100          # donations per run
N_REC = 30           # recipient organisations
RUNS = 50
W0 = dict(dist=0.25, urg=0.30, qty=0.20, cap=0.10, pick=0.05 * 2, cat=0.05)  # pick=0.10
W0["pick"] = 0.10
RMAX = 25.0
TMAX = 8.0           # hours
CATS = ["cooked", "packaged", "bakery", "produce"]

def gen(rng):
    rec = []
    for _ in range(N_REC):
        rec.append(dict(
            xy=rng.uniform(0, AREA, 2),
            need=rng.integers(40, 200),
            cap=rng.integers(20, 150),
            speed=rng.choice([15.0, 25.0, 35.0]),      # km/h effective
            cats=set(rng.choice(CATS, size=rng.integers(2, 5), replace=False)),
            accept=rng.uniform(0.75, 0.98)))
    don = []
    for _ in range(N_DON):
        c = rng.choice(CATS, p=[0.45, 0.25, 0.15, 0.15])
        tmax = {"cooked": 4.0, "bakery": 8.0, "packaged": 48.0, "produce": 24.0}[c]
        don.append(dict(xy=rng.uniform(0, AREA, 2), q=int(rng.integers(10, 80)),
                        cat=c, rem=float(rng.uniform(0.5, min(tmax, 8.0)))))
    return rec, don

def scores(d, r, state, now_need):
    dist = float(np.linalg.norm(d["xy"] - r["xy"]))
    s_dist = max(0.0, 1 - dist / RMAX)
    s_urg = 1 - d["rem"] / TMAX          # same for every recipient of a donation
    s_qty = min(1.0, d["q"] / max(now_need, 1))
    alloc = min(d["q"], now_need, r["cap"])
    s_cap = min(1.0, r["cap"] / max(alloc, 1))
    t_p = dist / r["speed"]
    s_pick = max(0.0, 1 - t_p / d["rem"])
    s_cat = 1.0
    return dist, dict(dist=s_dist, urg=s_urg, qty=s_qty, cap=s_cap, pick=s_pick, cat=s_cat), alloc, t_p

def eligible(d, r, now_need, dist, t_p):
    return (d["cat"] in r["cats"] and now_need > 0 and dist <= RMAX and t_p <= d["rem"])

def run(method, rec, don, rng_out, W=None):
    need = [r["need"] for r in rec]
    tot = sum(d["q"] for d in don)
    red = 0.0; ok = 0; dist_sum = 0.0; t_sum = 0.0; dl_ok = 0; att = 0; partial = 0
    for d in don:
        best = None; bs = -1.0
        for j, r in enumerate(rec):
            dist, sc, alloc, t_p = scores(d, r, None, need[j])
            if method == "nearest":
                # distance only, no eligibility check on time/need (a plain proximity app)
                val = -dist
            elif method == "urgency":
                if d["cat"] not in r["cats"] or need[j] <= 0: continue
                # urgent items go to the closest recipient that can arrive in time
                if t_p > d["rem"]: continue
                val = -dist
            else:
                if not eligible(d, r, need[j], dist, t_p): continue
                val = sum(W[k] * sc[k] for k in W)
            if best is None or val > bs:
                best, bs = j, val
        if best is None:
            continue
        r = rec[best]
        dist, sc, alloc, t_p = scores(d, r, None, need[best])
        att += 1
        if d["cat"] not in r["cats"] or need[best] <= 0:
            continue                      # recipient cannot use it: rejected
        # stochastic reality: traffic/volunteer delay and recipient acceptance
        delay = rng_out.lognormal(mean=-1.2, sigma=0.6)      # hours of extra delay
        t_real = t_p + delay
        accepted = rng_out.random() < r["accept"]
        in_time = t_real <= d["rem"]
        if accepted and in_time:
            ok += 1; red += alloc; need[best] -= alloc
            dist_sum += dist; t_sum += t_real; dl_ok += 1
            if alloc < d["q"]: partial += 1
    return dict(MSR=100 * ok / N_DON, FUR=100 * red / tot, ATD=dist_sum / max(ok, 1),
                APT=60 * t_sum / max(ok, 1), DCR=100 * dl_ok / max(att, 1), ATT=100 * att / N_DON)

def main():
    res = {m: [] for m in ["nearest", "urgency", "weighted"]}
    sens = {}
    variants = {
        "Base (0.25/0.30/0.20/0.10/0.10/0.05)": W0,
        "Equal weights": {k: 1 / 6 for k in W0},
        "Distance-heavy": dict(dist=0.45, urg=0.15, qty=0.15, cap=0.10, pick=0.10, cat=0.05),
        "Urgency-heavy": dict(dist=0.15, urg=0.45, qty=0.15, cap=0.10, pick=0.10, cat=0.05),
        "Quantity-heavy": dict(dist=0.15, urg=0.15, qty=0.40, cap=0.10, pick=0.15, cat=0.05),
        "Pickup-heavy": dict(dist=0.15, urg=0.15, qty=0.15, cap=0.10, pick=0.40, cat=0.05),
    }
    for k in variants: sens[k] = []
    perturb = []
    for seed in range(RUNS):
        rng = np.random.default_rng(seed)
        rec, don = gen(rng)
        for m in ["nearest", "urgency"]:
            res[m].append(run(m, rec, don, np.random.default_rng(1000 + seed)))
        res["weighted"].append(run("weighted", rec, don, np.random.default_rng(1000 + seed), W0))
        for k, W in variants.items():
            sens[k].append(run("weighted", rec, don, np.random.default_rng(1000 + seed), W))
        # random +-20% perturbation of base weights (renormalised)
        pr = np.random.default_rng(5000 + seed)
        W = {k: v * pr.uniform(0.8, 1.2) for k, v in W0.items()}
        s = sum(W.values()); W = {k: v / s for k, v in W.items()}
        perturb.append(run("weighted", rec, don, np.random.default_rng(1000 + seed), W))
    def agg(lst):
        out = {}
        for k in lst[0]:
            a = np.array([x[k] for x in lst]); out[k] = (float(a.mean()), float(a.std()))
        return out
    out = dict(main={m: agg(v) for m, v in res.items()},
               sens={k: agg(v) for k, v in sens.items()},
               perturb=agg(perturb),
               perturb_range={k: (float(min(x[k] for x in perturb)), float(max(x[k] for x in perturb))) for k in perturb[0]},
               cfg=dict(N_DON=N_DON, N_REC=N_REC, RUNS=RUNS, AREA=AREA))
    json.dump(out, open("sim_results.json", "w"), indent=1)
    for m, v in out["main"].items():
        print(m, {k: f"{a:.1f}+-{b:.1f}" for k, (a, b) in v.items()})
    print("--- sensitivity")
    for m, v in out["sens"].items():
        print(m, {k: f"{a:.1f}" for k, (a, b) in v.items()})
    print("perturb", {k: f"{a:.1f}+-{b:.1f}" for k, (a, b) in out["perturb"].items()})
    print("range", {k: (round(a, 1), round(b, 1)) for k, (a, b) in out["perturb_range"].items()})

if __name__ == "__main__":
    main()
