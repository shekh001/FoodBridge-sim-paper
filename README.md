# Surplus food redistribution: matching simulation

Simulation code for the paper *Multi-Factor Matching with Re-Matching and Partial Allocation for Surplus Food Redistribution* (School of Engineering and Technology, K.R. Mangalam University).

All data is **synthetic**. The results show how the matching rules behave in a simple model. They are not evidence about real donors or NGOs.

## What it compares

1. Nearest recipient (no checks)
2. Urgency-aware nearest (category, remaining need, arrival before deadline)
3. Weighted score only (six-factor score, top recipient)
4. Weighted score + re-matching (next-ranked recipient after a decline or delay)
5. Full system (re-matching + partial allocation to up to three recipients)

Metrics: matching success rate (MSR), food utilisation rate (FUR), average transport distance (ATD), average pickup time (APT).

## Run

```
pip install -r requirements.txt
python3 sim2.py
```

Runs take well under a minute. Random seeds are fixed (50 runs), so the output is the same on every run. The expected output is in `expected_output.txt`; the numbers in Table II of the paper come from it.

## Files

- `sim.py`: scenario generator, factor scores, eligibility rules, the three simple approaches, and a first weight-sensitivity run
- `sim2.py`: full system with re-matching and partial allocation, the weight sensitivity test (five weight settings and random +/-20% changes), and the paired comparison; writes `sim2_results.json`

## Main settings

20 x 20 km area, 30 recipients, 100 donations per run, 50 runs. Recipient need 40-199 meals, capacity 20-149, speed 15/25/35 km/h, acceptance probability 0.75-0.98. Donations 10-79 meals. A random delay (log-normal, median about 18 minutes) is added to every pickup. Weights: distance 0.25, urgency 0.30, quantity 0.20, capacity 0.10, pickup 0.10, category 0.05.

## Limits

Synthetic data; the success rule uses the same quantities (time and need) as the score; food safety, fairness and real road networks are not modelled.

## AI use

Parts of this code were drafted with the help of a generative AI tool (Claude, Anthropic).
