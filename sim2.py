import json, numpy as np
import sim
from sim import gen, scores, eligible, W0, N_DON, RUNS

def run_full(rec, don, rng, W, rematch=True, partial=True, max_recip=3):
    need = [r["need"] for r in rec]
    tot = sum(d["q"] for d in don)
    red = 0.0; ok = 0; dsum = 0.0; tsum = 0.0; ndel = 0
    for d in don:
        ranked = []
        for j, r in enumerate(rec):
            dist, sc, alloc, t_p = scores(d, r, None, need[j])
            if eligible(d, r, need[j], dist, t_p):
                ranked.append((sum(W[k] * sc[k] for k in W), j))
        ranked.sort(reverse=True)
        remaining = d["q"]; used = 0; got = False
        for _, j in ranked:
            if remaining <= 0 or used >= max_recip: break
            r = rec[j]
            dist, sc, alloc, t_p = scores(d, r, None, need[j])
            alloc = min(remaining, need[j], r["cap"])
            if alloc <= 0: continue
            t_real = t_p + rng.lognormal(-1.2, 0.6)
            accepted = rng.random() < r["accept"]
            if accepted and t_real <= d["rem"]:
                red += alloc; need[j] -= alloc; remaining -= alloc; used += 1; got = True
                dsum += dist; tsum += t_real; ndel += 1
                if not partial: break
            else:
                if not rematch: break
        if got: ok += 1
    return dict(MSR=100 * ok / N_DON, FUR=100 * red / tot, ATD=dsum / max(ndel, 1), APT=60 * tsum / max(ndel, 1))

def agg(lst):
    return {k: (float(np.mean([x[k] for x in lst])), float(np.std([x[k] for x in lst]))) for k in lst[0]}

methods = {"nearest": [], "urgency": [], "weighted": [], "weighted_rematch": [], "weighted_full": []}
sens = {}
variants = {
    "Base weights": W0,
    "Equal weights": {k: 1/6 for k in W0},
    "Distance-heavy": dict(dist=0.45, urg=0.15, qty=0.15, cap=0.10, pick=0.10, cat=0.05),
    "Urgency-heavy": dict(dist=0.15, urg=0.45, qty=0.15, cap=0.10, pick=0.10, cat=0.05),
    "Quantity-heavy": dict(dist=0.15, urg=0.15, qty=0.40, cap=0.10, pick=0.15, cat=0.05),
    "Pickup-heavy": dict(dist=0.15, urg=0.15, qty=0.15, cap=0.10, pick=0.40, cat=0.05),
}
for k in variants: sens[k] = []
pert = []
for seed in range(RUNS):
    rec, don = gen(np.random.default_rng(seed))
    R = lambda: np.random.default_rng(1000 + seed)
    for m in ["nearest", "urgency"]:
        r = sim.run(m, rec, don, R()); methods[m].append({k: r[k] for k in ["MSR","FUR","ATD","APT"]})
    r = sim.run("weighted", rec, don, R(), W0); methods["weighted"].append({k: r[k] for k in ["MSR","FUR","ATD","APT"]})
    methods["weighted_rematch"].append(run_full(rec, don, R(), W0, True, False))
    methods["weighted_full"].append(run_full(rec, don, R(), W0, True, True))
    for k, W in variants.items(): sens[k].append(run_full(rec, don, R(), W, True, True))
    pr = np.random.default_rng(5000 + seed)
    W = {k: v * pr.uniform(0.8, 1.2) for k, v in W0.items()}; s = sum(W.values()); W = {k: v / s for k, v in W.items()}
    pert.append(run_full(rec, don, R(), W, True, True))
out = dict(main={m: agg(v) for m, v in methods.items()}, sens={k: agg(v) for k, v in sens.items()},
           pert=agg(pert), pert_min={k: float(min(x[k] for x in pert)) for k in pert[0]},
           pert_max={k: float(max(x[k] for x in pert)) for k in pert[0]})
# paired comparison full vs urgency on FUR
d = np.array([a["FUR"] - b["FUR"] for a, b in zip(methods["weighted_full"], methods["urgency"])])
out["paired_FUR_full_minus_urgency"] = (float(d.mean()), float(d.std() / np.sqrt(len(d))), int((d > 0).sum()), len(d))
json.dump(out, open("sim2_results.json", "w"), indent=1)
for m, v in out["main"].items(): print(m, {k: f"{a:.1f}±{b:.1f}" for k, (a, b) in v.items()})
print("sens"); [print(" ", m, {k: f"{a:.1f}" for k, (a, b) in v.items()}) for m, v in out["sens"].items()]
print("pert", {k: f"{a:.1f}±{b:.1f}" for k, (a, b) in out["pert"].items()}, out["pert_min"], out["pert_max"])
print("paired", out["paired_FUR_full_minus_urgency"])
