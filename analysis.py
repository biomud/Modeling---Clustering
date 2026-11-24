"""Reproduce the gym clustering assignment using training-only model fitting."""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
FEATURES = [
    "age", "visits_per_month", "avg_visit_minutes", "classes_per_month",
    "months_member", "monthly_extras_spend",
]
SEED = 42
CHOSEN_K = 4  # Training elbow: improvement becomes much smaller after four groups.
CREDIT = 10
SPEND_THRESHOLD = 20
ADDITIONAL_SPEND = 25
MARGIN = 0.60
NEW_MEMBER = dict(zip(FEATURES, [30, 4, 42, 1, 5, 10]))
# Names are ordered by increasing visits/month in the fitted training centroids.
SEGMENT_NAMES = [
    "New occasional visitors", "Class-focused spenders",
    "Long-tenure regulars", "Frequent independent users",
]


def credit_net(spending):
    """Incremental monthly margin relative to giving no credit; $20 is high spend."""
    return np.where(np.asarray(spending) < SPEND_THRESHOLD,
                    ADDITIONAL_SPEND * MARGIN - CREDIT, -CREDIT)


@dataclass
class GymSegments:
    scaler: StandardScaler
    kmeans: KMeans
    names: dict
    gives_credit: dict

    def offer_for_member(self, member):
        """Take one member's six features and return their segment and credit decision."""
        row = pd.DataFrame([{feature: member[feature] for feature in FEATURES}])
        if not np.isfinite(row.to_numpy(dtype=float)).all():
            raise ValueError("All six member features must be finite numbers.")
        cluster = int(self.kmeans.predict(self.scaler.transform(row))[0])
        return {"segment": self.names[cluster],
                "gets_credit": bool(self.gives_credit[cluster])}


def fit_analysis(data):
    """Fit all candidate models and the credit policy using only the 70% training set."""
    if list(data.columns) != ["member_id", *FEATURES] or data.shape != (330, 7):
        raise ValueError("Expected the assignment's 330-row, seven-column dataset.")
    if data.member_id.duplicated().any() or not np.isfinite(data[FEATURES]).all().all():
        raise ValueError("Member IDs must be unique and all features must be finite.")
    train, test = train_test_split(data, test_size=0.30, random_state=SEED)
    scaler = StandardScaler().fit(train[FEATURES])
    train_x, test_x = scaler.transform(train[FEATURES]), scaler.transform(test[FEATURES])
    models, rows = {}, []
    for k in range(1, 9):
        model = KMeans(n_clusters=k, n_init=50, random_state=SEED, algorithm="lloyd")
        model.fit(train_x)
        models[k] = model
        # transform returns Euclidean distances; square before summing nearest distances.
        test_sse = float(np.square(model.transform(test_x)).min(axis=1).sum())
        rows.append({"k": k, "train_sse": model.inertia_, "test_sse": test_sse,
                     "train_sse_per_member": model.inertia_ / len(train),
                     "test_sse_per_member": test_sse / len(test)})
    metrics = pd.DataFrame(rows)
    metrics["train_reduction_from_previous_pct"] = -metrics.train_sse.pct_change() * 100
    metrics["test_reduction_from_previous_pct"] = -metrics.test_sse.pct_change() * 100

    selected = models[CHOSEN_K]
    centers = pd.DataFrame(scaler.inverse_transform(selected.cluster_centers_), columns=FEATURES)
    ordered_clusters = centers.sort_values("visits_per_month").index.tolist()
    names = dict(zip(ordered_clusters, SEGMENT_NAMES))
    # Freeze the offer policy from individual TRAINING member economics, not centroids.
    train_net = credit_net(train.monthly_extras_spend)
    policy = {cluster: bool(train_net[selected.labels_ == cluster].sum() > 0)
              for cluster in ordered_clusters}
    fitted = GymSegments(scaler, selected, names, policy)

    assignments = data.copy()
    assignments["split"] = np.where(data.index.isin(train.index), "train", "test")
    assignments["cluster_id"] = selected.predict(scaler.transform(data[FEATURES]))
    assignments["segment"] = assignments.cluster_id.map(names)
    assignments["gets_credit"] = assignments.cluster_id.map(policy)
    assignments["extras_below_20"] = data.monthly_extras_spend < SPEND_THRESHOLD
    assignments["baseline_net"] = credit_net(data.monthly_extras_spend).astype(int)
    assignments["v1_net"] = np.where(assignments.gets_credit, assignments.baseline_net, 0)

    centers.insert(0, "cluster_id", centers.index)
    centers.insert(1, "segment", centers.cluster_id.map(names))
    centers["gets_credit"] = centers.cluster_id.map(policy)
    summary_rows = []
    for cluster in ordered_clusters:
        group = assignments[assignments.cluster_id == cluster]
        training = group[group.split == "train"]
        summary_rows.append({
            "segment": names[cluster], "train_count": len(training),
            "test_count": int((group.split == "test").sum()), "all_count": len(group),
            "train_below_20": int(training.extras_below_20.sum()),
            "train_below_20_share": training.extras_below_20.mean(),
            "train_net_if_credit": int(training.baseline_net.sum()),
            "all_below_20": int(group.extras_below_20.sum()),
            "all_at_least_20": int((~group.extras_below_20).sum()),
            "all_mean_extras": group.monthly_extras_spend.mean(),
            "gets_credit": policy[cluster],
            "all_baseline_net": int(group.baseline_net.sum()),
            "all_v1_net": int(group.v1_net.sum()),
        })
    summary = pd.DataFrame(summary_rows)
    finance = []
    for split, group in [("all", assignments), ("train", assignments[assignments.split == "train"]),
                         ("test", assignments[assignments.split == "test"])]:
        recipients = group[group.gets_credit]
        finance.append({
            "population": split, "members": len(group),
            "baseline_recipients": len(group), "v1_recipients": len(recipients),
            "baseline_credit_cost": int(len(group) * CREDIT),
            "v1_credit_cost": int(len(recipients) * CREDIT),
            "baseline_incremental_margin": int(group.extras_below_20.sum() * ADDITIONAL_SPEND * MARGIN),
            "v1_incremental_margin": int(recipients.extras_below_20.sum() * ADDITIONAL_SPEND * MARGIN),
            "baseline_net": int(group.baseline_net.sum()), "v1_net": int(group.v1_net.sum()),
            "improvement": int(group.v1_net.sum() - group.baseline_net.sum()),
        })
    return fitted, metrics, assignments, centers.loc[ordered_clusters], summary, pd.DataFrame(finance)


def make_charts(metrics, finance, output):
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.7), layout="constrained")
    for ax, split, count, color in zip(axes, ["train", "test"], [231, 99], ["#2563a6", "#b45f24"]):
        values = metrics[f"{split}_sse"]
        ax.plot(metrics.k, values, "o-", color=color, linewidth=2)
        ax.axvline(CHOSEN_K, color="#777777", linestyle="--", linewidth=1)
        for k in [1, CHOSEN_K]:
            y = values.iloc[k - 1]
            ax.annotate(f"{y:,.1f}", (k, y), xytext=(7, 10), textcoords="offset points")
        ax.set(title=f"{split.title()} members (n = {count})", xlabel="Number of segments (k)",
               ylabel="Within-group spread (sum of squared distances)")
        ax.set_xticks(range(1, 9))
        ax.set_xlim(0.7, 8.4)
        ax.set_ylim(0, values.max() * 1.16)
        ax.grid(axis="y", alpha=0.18)
    fig.suptitle("Elbow comparison: training and held-out members", fontsize=16)
    fig.savefig(output / "elbow.png", dpi=220)
    plt.close(fig)

    overall = finance.iloc[0]
    fig, ax = plt.subplots(figsize=(7, 4.4), layout="constrained")
    values = [overall.baseline_net, overall.v1_net]
    bars = ax.bar(["Baseline: credit to all", "v1: selected segment"], values,
                  color=["#767676", "#2563a6"], width=0.5)
    for bar, value in zip(bars, values):
        ax.annotate(f"{'−' if value < 0 else '+'}${abs(value):,}",
                    (bar.get_x() + bar.get_width()/2, value),
                    xytext=(0, -18 if value < 0 else 8), textcoords="offset points",
                    ha="center", fontsize=14)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set(title="Monthly net from the credit: all 330 members", ylabel="Incremental monthly net ($)")
    ax.set_ylim(min(values) - 280, max(values) + 240)
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    fig.savefig(output / "credit_value.png", dpi=220)
    plt.close(fig)


def md_table(frame):
    """Small Markdown table without an additional formatting dependency."""
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.2f}" if np.isfinite(value) else "—"
        return str(value)
    rows = ["| " + " | ".join(frame.columns) + " |",
            "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    rows.extend("| " + " | ".join(cell(v) for v in row) + " |"
                for row in frame.itertuples(index=False, name=None))
    return "\n".join(rows)


def main():
    source = ROOT / "gym_members.csv"
    data = pd.read_csv(source)
    fitted, metrics, assignments, centers, summary, finance = fit_analysis(data)
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    for name, frame in [("fit_metrics", metrics), ("member_assignments", assignments),
                        ("training_centroids", centers), ("segment_summary", summary),
                        ("monthly_value", finance)]:
        frame.to_csv(output / f"{name}.csv", index=False, float_format="%.10f")
    prediction = fitted.offer_for_member(NEW_MEMBER)
    baseline, selected = metrics.iloc[0], metrics.iloc[CHOSEN_K - 1]
    reduction = 100 * (1 - selected.test_sse / baseline.test_sse)
    result = {
        "data_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "rows": len(data), "mean_monthly_extras": data.monthly_extras_spend.mean(),
        "features": FEATURES, "excluded": ["member_id"], "random_seed": SEED,
        "k": CHOSEN_K, "n_init": 50, "train_rows": 231, "test_rows": 99,
        "test_sse_baseline": baseline.test_sse, "test_sse_v1": selected.test_sse,
        "test_spread_reduction_pct": reduction,
        "new_member_features": NEW_MEMBER, "new_member_decision": prediction,
        "monthly_economics": finance.to_dict(orient="records"),
        "training_scaler_mean": fitted.scaler.mean_.tolist(),
        "training_scaler_scale": fitted.scaler.scale_.tolist(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                     "matplotlib": matplotlib.__version__},
    }
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    make_charts(metrics, finance, output)
    traits = centers[["segment", *FEATURES]].set_index("segment").T.reset_index(names="Training centroid feature")
    overall = finance.iloc[0]
    report = f"""# Gym member clustering: technical analysis

## Data and model

Source: the course-provided `gym_members.csv`. {len(data)} members, seven columns,
with usage/spending averages over the previous three months. Exact calendar dates
are not supplied. Mean monthly extras spend: ${data.monthly_extras_spend.mean():.2f}.
There are no missing feature values or duplicate member IDs.

Random 70/30 split with seed {SEED}: 231 training members and 99 test members.
Use all six numeric features except `member_id`, which identifies a row rather than
describing a member. Age describes demographics, months_member describes tenure,
and the remaining features describe usage and spending. StandardScaler uses only
training means and population standard deviations. K-means uses Euclidean distances,
k-means++ initialization, 50 initializations, Lloyd's algorithm, and seed {SEED}.
All candidate models share the same features, split, scaling, and settings; only k changes.

## Fit comparison

Choose k = {CHOSEN_K} from the training elbow. The training spread falls substantially
through k = 4, then decreases much more slowly. The held-out curve has the same bend.
Test members are assigned to training centers; neither scaling nor centers are refitted.
The model is also not refitted when assigning all 330 members.

{md_table(metrics)}

Test spread: {baseline.test_sse:.3f} at k = 1 versus {selected.test_sse:.3f} at
k = {CHOSEN_K}, a {reduction:.2f}% reduction. These are squared distances in standardized
feature space, not dollars or a classification accuracy. Raw training/test sums differ
partly because sample sizes differ; per-member values are included for comparison.

![Training and test elbow curves](elbow.png)

## Training centroids in original units

Names describe relative behavior in this dataset, not verified motivations.
Full-population counts below are distinct from the training-only centroids here.

{md_table(traits)}

- New occasional visitors: low visit frequency, short tenure, and low extras spending.
- Class-focused spenders: frequent classes and the highest extras spending.
- Long-tenure regulars: older members with long tenure and moderate extras spending.
- Frequent independent users: many long visits and few group classes.

## Segment credit policy

Finance assigns +$5 net to a recipient spending below $20 and -$10 to a recipient
spending at least $20. No credit means $0 incremental net. For each segment, sum
these member-level values on the training members and offer the credit only if the
sum is positive. Equivalently, more than two thirds must be below $20. A tie gets no
credit. The policy is fixed before applying it to test members or the full population.
Do not decide from centroid spending: a mean below $20 does not imply positive net.

{md_table(summary)}

## New member

Input: {json.dumps(NEW_MEMBER)}.
The fitted `GymSegments.offer_for_member(member)` function returns
`{json.dumps(prediction)}`. It uses the existing scaler, nearest training center,
and fixed segment policy. This first-month use assumes the supplied measurements
are comparable to the three-month averages used in training; the data do not test
that temporal assumption.

## Monthly value

The baseline sends the credit to everyone. It is a single targeting group, but its
economics must still be calculated member by member, not from the overall $26.01 mean.
These values are incremental margin from the credit relative to no credit, not total
gym revenue/profit. Under the supplied assumptions:

{md_table(finance)}

For all members, baseline = ${int(overall.baseline_incremental_margin):,} added margin
minus ${int(overall.baseline_credit_cost):,} credits = ${int(overall.baseline_net):,}/month.
v1 = ${int(overall.v1_incremental_margin):,} added margin minus
${int(overall.v1_credit_cost):,} credits = ${int(overall.v1_net):,}/month.
Difference = ${int(overall.improvement):,}/month. The full-population calculation includes
both training and test members as requested; it is not an entirely held-out estimate.
The separate test row evaluates the fixed policy on held-out members.
Finance's response assumptions are stipulated, not estimated or causally validated here.

![Monthly credit net](credit_value.png)

## Reproducibility

Run `python analysis.py` from an environment with `requirements.txt` installed.
The script writes all tables, figures, this report, and `results.json`; no slide
values need to be typed into the analysis. Source SHA-256: `{result['data_sha256']}`.
Installed versions are recorded in `results.json`.

This is a technical analysis, not submission slides. The assignment requires the
student to write the slides: one title slide plus two content slides, exported to
one PDF. The title slide needs the student's name and the repository link.
"""
    (output / "analysis.md").write_text(report)
    print(json.dumps({"chosen_k": CHOSEN_K, "test_spread_reduction_pct": reduction,
                      "new_member": prediction, "monthly_value": finance.iloc[0].to_dict()}, indent=2))


if __name__ == "__main__":
    main()
