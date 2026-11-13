#!/usr/bin/env python3
"""Train and compare Logistic Regression vs. SVM (RBF) URL classifiers.

Reads a CSV with `url` and `label` columns (default: data/urldata.csv),
extracts the six features from features.py, trains both models inside
StandardScaler pipelines on an 80% split, evaluates on the 20% hold-out and
saves:

    models/lr_model.pkl      Logistic Regression pipeline
    models/svm_model.pkl     SVC(kernel="rbf", probability=True) pipeline
    models/metrics.pkl       metrics + confusion matrices + dataset metadata

Class convention: 1 = malicious (positive class), 0 = legitimate.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features import FEATURE_NAMES, features_dataframe

# SVC(probability=True) is required by this project; newer scikit-learn only warns that it is deprecated.
warnings.filterwarnings("ignore", message=".*`probability` parameter was deprecated.*", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = BASE_DIR / "data" / "urldata.csv"
MODELS_DIR = BASE_DIR / "models"

MALICIOUS_LABELS = {"malicious", "bad", "phishing", "malware", "defacement", "spam", "unsafe", "1", "true"}
LEGIT_LABELS = {"legitimate", "legit", "benign", "good", "safe", "0", "false"}
LABEL_COLUMN_FALLBACKS = ("type", "status", "result", "class")

MODEL_FILES = {"Logistic Regression": "lr_model.pkl", "SVM": "svm_model.pkl"}


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"[error] dataset not found: {path}\n        run: python generate_dataset.py")
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "url" not in df.columns:
        sys.exit("[error] CSV must contain a 'url' column.")
    label_col = "label" if "label" in df.columns else next(
        (c for c in LABEL_COLUMN_FALLBACKS if c in df.columns), None)
    if label_col is None:
        sys.exit("[error] CSV must contain a 'label' column.")

    df = df[["url", label_col]].dropna().rename(columns={label_col: "label"})
    df["url"] = df["url"].astype(str).str.strip()
    df = df[df["url"] != ""].drop_duplicates(subset="url")

    if pd.api.types.is_numeric_dtype(df["label"]):
        df["label"] = df["label"].astype(int)
    norm = df["label"].astype(str).str.strip().str.lower()

    unknown = sorted(set(norm) - MALICIOUS_LABELS - LEGIT_LABELS)
    if unknown:
        sys.exit(f"[error] unrecognised label values: {unknown}. "
                 f"Use e.g. legitimate/malicious, benign/phishing, good/bad or 0/1.")
    df["y"] = norm.isin(MALICIOUS_LABELS).astype(int)
    return df.reset_index(drop=True)


def build_models(seed: int) -> dict:
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=seed)),
        ]),
    }


def evaluate(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        # [[TN, FP], [FN, TP]]  (rows = actual, columns = predicted; 1 = malicious)
        "confusion_matrix": confusion_matrix(y_test, pred, labels=[0, 1]).astype(int).tolist(),
    }


def print_report(results: dict) -> None:
    line = "+" + "+".join("-" * w for w in (24, 12, 13, 10, 12)) + "+"
    print("\n" + line)
    print(f"| {'Model':<22} | {'Accuracy':>9}  | {'Precision':>10}  | {'Recall':>7}  | {'F1-Score':>9}  |")
    print(line)
    for name, m in results.items():
        print(f"| {name:<22} | {m['accuracy']:>9.4f}  | {m['precision']:>10.4f}  | "
              f"{m['recall']:>7.4f}  | {m['f1']:>9.4f}  |")
    print(line)

    for name, m in results.items():
        (tn, fp), (fn, tp) = m["confusion_matrix"]
        print(f"\nConfusion matrix: {name}   (rows = actual, columns = predicted)")
        print(f"                       Pred legitimate   Pred malicious")
        print(f"  Actual legitimate    {tn:>15}   {fp:>14}")
        print(f"  Actual malicious     {fn:>15}   {tp:>14}")


def verify_artifacts(X_test, expected_preds: dict) -> None:
    """Reload every saved .pkl and confirm it behaves identically to the in-memory model."""
    print("\n[verify] reloading saved artifacts ...")
    for name, filename in MODEL_FILES.items():
        path = MODELS_DIR / filename
        model = joblib.load(path)
        reloaded = model.predict(X_test)
        proba = model.predict_proba(X_test)
        assert np.array_equal(reloaded, expected_preds[name]), f"{filename}: predictions differ after reload"
        assert proba.shape == (len(X_test), 2) and np.allclose(proba.sum(axis=1), 1.0), f"{filename}: bad probabilities"
        print(f"  [OK] {filename:<14} {path.stat().st_size / 1024:>7.1f} KB  predict/predict_proba match")

    metrics = joblib.load(MODELS_DIR / "metrics.pkl")
    assert set(metrics["models"]) == set(MODEL_FILES), "metrics.pkl is missing a model"
    for name, m in metrics["models"].items():
        assert {"accuracy", "precision", "recall", "f1", "confusion_matrix"} <= set(m), f"metrics.pkl incomplete for {name}"
    print(f"  [OK] metrics.pkl      models={list(metrics['models'])}  "
          f"train={metrics['meta']['train_samples']} test={metrics['meta']['test_samples']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DEFAULT_DATA), help="CSV with url,label columns")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--max-samples", type=int, default=None,
                    help="optional stratified subsample for very large CSVs (SVM training is O(n^2))")
    args = ap.parse_args()

    data_path = Path(args.data)
    df = load_dataset(data_path)
    if args.max_samples and len(df) > args.max_samples:
        df, _ = train_test_split(df, train_size=args.max_samples, random_state=args.seed, stratify=df["y"])
        df = df.reset_index(drop=True)

    n_mal = int(df["y"].sum())
    print(f"[data] {data_path}  rows={len(df)}  legitimate={len(df) - n_mal}  malicious={n_mal}")
    if min(n_mal, len(df) - n_mal) < 10:
        sys.exit("[error] need at least 10 samples of each class.")

    print("[features] extracting:", ", ".join(FEATURE_NAMES))
    X = features_dataframe(df["url"])
    y = df["y"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y)
    print(f"[split] train={len(X_train)}  test={len(X_test)}  (stratified, seed={args.seed})")

    results, preds = {}, {}
    for name, model in build_models(args.seed).items():
        print(f"[train] {name} ...", end=" ", flush=True)
        model.fit(X_train, y_train)
        results[name] = evaluate(model, X_test, y_test)
        preds[name] = model.predict(X_test)
        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(model, MODELS_DIR / MODEL_FILES[name])
        print("done")

    print_report(results)

    metrics = {
        "models": results,
        "meta": {
            "dataset": str(data_path.name),
            "total_samples": int(len(df)),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "class_balance": {"legitimate": int(len(df) - n_mal), "malicious": n_mal},
            "features": FEATURE_NAMES,
            "test_size": float(args.test_size),
            "seed": int(args.seed),
            "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "sklearn_version": sklearn.__version__,
        },
    }
    joblib.dump(metrics, MODELS_DIR / "metrics.pkl")
    verify_artifacts(X_test, preds)
    print("\n[done] artifacts saved in", MODELS_DIR)


if __name__ == "__main__":
    main()
