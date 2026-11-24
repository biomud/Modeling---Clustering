# Gym member clustering

Code and technical analysis for the gym monthly-credit assignment.

Repository: https://github.com/biomud/Modeling---Clustering

## Reproduce

The project was run with Python 3.12.9 and the package versions in `requirements.txt`.
Using the existing Anaconda installation on this computer:

```bash
/opt/anaconda3/bin/python analysis.py
```

On another computer, create an environment and run:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python analysis.py
```

The script preserves the input data and regenerates files in `results/`.

## Read the results

- [Technical analysis](results/analysis.md): data, model specification, selection of k,
  centroids, segment policy, new-member prediction, and member-level economics.
- [Elbow curves](results/elbow.png): training and test spread for k = 1 to 8.
- [Credit value chart](results/credit_value.png): monthly net for the two plans.
- [Member assignments](results/member_assignments.csv): every member's split,
  cluster, segment, credit decision, and contribution to each plan.
- [Training centroids](results/training_centroids.csv),
  [segment summary](results/segment_summary.csv),
  [fit metrics](results/fit_metrics.csv), and [monthly value](results/monthly_value.csv).
- [Machine-readable results](results/results.json): key numbers, new-member result,
  input checksum, scaler parameters, and software versions.

## Score one member

```python
import pandas as pd
from analysis import fit_analysis

fitted, *_ = fit_analysis(pd.read_csv("gym_members.csv"))
decision = fitted.offer_for_member({
    "age": 30,
    "visits_per_month": 4,
    "avg_visit_minutes": 42,
    "classes_per_month": 1,
    "months_member": 5,
    "monthly_extras_spend": 10,
})
print(decision)
```

Fit once, then reuse `fitted.offer_for_member(...)` for new members. The function
does not refit the scaler, centers, or credit policy when scoring a member.

## Submission requirement

The assignment permits AI-assisted code and code-generated charts, but requires
student-authored slides. This repository supplies the analysis and figures, not a
finished slide submission. Submit one PDF with a title slide (your name and this
repository link) followed by two content slides. See the original
[assignment](Clustering%20assignment.md) for the required coverage.
