#!/usr/bin/env python3
"""Flask backend for the SOC URL Threat Detection dashboard.

Routes
------
GET  /                 dashboard UI
POST /api/scan         classify a URL        {"url": "...", "model": "svm" | "lr", "log": true}
GET  /api/metrics      model comparison metrics (from models/metrics.pkl)
GET  /api/logs         this session's SIEM incident log
POST /api/logs/clear   clear this session's log
GET  /api/health       model / metrics availability

Run:  python app.py      (then open http://127.0.0.1:5000)
"""
from __future__ import annotations

import os
import re
import secrets
import threading
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request, session

from features import FEATURE_NAMES, extract_features, hostname_of

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

MODEL_REGISTRY = {
    "svm": {"label": "SVM", "file": "svm_model.pkl"},
    "lr": {"label": "Logistic Regression", "file": "lr_model.pkl"},
}
MODEL_ALIASES = {
    "svm": "svm", "svc": "svm",
    "lr": "lr", "logreg": "lr", "logistic": "lr",
    "logistic regression": "lr", "logistic_regression": "lr",
}

CRITICAL_THRESHOLD = 80.0   # risk score >= 80       -> CRITICAL ALERT
HIGH_THRESHOLD = 50.0       # 50 <= risk score < 80  -> HIGH WARNING; < 50 -> INFO / SAFE
MAX_URL_LEN = 2048
MAX_LOG_ENTRIES = 200
MAX_SESSIONS = 500

app = Flask(__name__)
app.json.sort_keys = False
app.config.update(
    SECRET_KEY=os.environ.get("SECRET_KEY") or secrets.token_hex(32),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    PERMANENT_SESSION_LIFETIME=timedelta(hours=8),
)

# --------------------------------------------------------------------------- #
# Model / metrics loading (retried on demand so you can train while app runs)
# --------------------------------------------------------------------------- #
_artifact_lock = threading.Lock()
_models: dict = {}
_metrics: dict | None = None


def load_artifacts() -> None:
    global _metrics
    with _artifact_lock:
        for key, spec in MODEL_REGISTRY.items():
            path = MODELS_DIR / spec["file"]
            if key not in _models and path.exists():
                _models[key] = joblib.load(path)
        if _metrics is None and (MODELS_DIR / "metrics.pkl").exists():
            _metrics = joblib.load(MODELS_DIR / "metrics.pkl")


def get_model(key: str):
    if key not in _models:
        load_artifacts()
    return _models.get(key)


load_artifacts()

# --------------------------------------------------------------------------- #
# Per-session SIEM log (server-side; the cookie only carries an opaque id)
# --------------------------------------------------------------------------- #
_log_lock = threading.Lock()
_log_store: "OrderedDict[str, list]" = OrderedDict()


def _session_id() -> str:
    sid = session.get("sid")
    if not sid:
        sid = secrets.token_hex(16)
        session["sid"] = sid
        session.permanent = True
    return sid


def add_log(entry: dict) -> None:
    sid = _session_id()
    with _log_lock:
        entries = _log_store.setdefault(sid, [])
        _log_store.move_to_end(sid)
        entries.insert(0, entry)
        del entries[MAX_LOG_ENTRIES:]
        while len(_log_store) > MAX_SESSIONS:
            _log_store.popitem(last=False)


def get_logs() -> list:
    with _log_lock:
        return list(_log_store.get(_session_id(), []))


def clear_logs() -> None:
    with _log_lock:
        _log_store.pop(_session_id(), None)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")


def normalize_url(raw: str) -> str:
    """Prefix bare hosts like 'github.com/x' with http:// (no scheme means no https)."""
    return raw if _SCHEME_RE.match(raw) else "http://" + raw


def resolve_model(value) -> str | None:
    return MODEL_ALIASES.get(str(value or "svm").strip().lower())


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def build_alert(score: float, host: str, has_ip: bool, url: str) -> dict:
    """Simulated automated SOC response for a given risk score."""
    now = _utc_now()
    target = host or url
    suffix = secrets.token_hex(2).upper()

    if score >= CRITICAL_THRESHOLD:
        ticket = f"INC-{now:%Y%m%d}-{suffix}"
        sinkhole = (
            f"Raw IP host, so no DNS record to poison. {target} null-routed at the edge router."
            if has_ip else
            f"{target} and its subdomains resolve to the internal sinkhole 10.255.255.53."
        )
        return {
            "level": "CRITICAL", "label": "CRITICAL ALERT", "severity": "critical",
            "summary": "High-confidence malicious URL. Automated containment executed.",
            "incident_id": ticket,
            "actions": [
                {"title": "Firewall block",
                 "detail": f"Deny rule pushed to perimeter firewalls for {target} (inbound and outbound)."},
                {"title": "DNS sinkhole", "detail": sinkhole},
                {"title": "SOC incident ticket",
                 "detail": f"{ticket} opened, priority P1, assigned to the Tier-2 SOC queue."},
            ],
            "log_action": f"Firewall block + DNS sinkhole + ticket {ticket}",
        }

    if score >= HIGH_THRESHOLD:
        triage = f"TRG-{now:%Y%m%d}-{suffix}"
        return {
            "level": "HIGH", "label": "HIGH WARNING", "severity": "high",
            "summary": "Suspicious URL. Contained for analyst review.",
            "incident_id": triage,
            "actions": [
                {"title": "Endpoint sandbox isolation",
                 "detail": f"Sessions to {target} redirected into an isolated browser sandbox."},
                {"title": "Analyst triage alert",
                 "detail": f"{triage} raised to the Tier-1 triage queue, priority P2."},
            ],
            "log_action": f"Sandbox isolation + analyst triage {triage}",
        }

    return {
        "level": "INFO", "label": "INFO / SAFE", "severity": "info",
        "summary": "No threat indicators above the alert threshold.",
        "incident_id": None,
        "actions": [
            {"title": "Traffic allowed", "detail": f"Connection to {target} permitted by policy."},
            {"title": "SIEM baseline", "detail": "Event written to the url_baseline index for trend analysis."},
        ],
        "log_action": "Traffic allowed, logged to SIEM baseline",
    }


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.after_request
def security_headers(resp):
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    resp.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; connect-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'",
    )
    if request.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/")
def dashboard():
    _session_id()
    return render_template("dashboard.html")


@app.get("/api/health")
def api_health():
    load_artifacts()
    return jsonify({
        "models_loaded": [MODEL_REGISTRY[k]["label"] for k in MODEL_REGISTRY if k in _models],
        "metrics_loaded": _metrics is not None,
        "ready": len(_models) == len(MODEL_REGISTRY),
    })


@app.get("/api/metrics")
def api_metrics():
    load_artifacts()
    if _metrics is None:
        return jsonify({"error": "models/metrics.pkl not found. Run: python train_models.py"}), 503
    return jsonify(_metrics)


@app.post("/api/scan")
def api_scan():
    payload = request.get_json(silent=True) or {}
    raw_url = str(payload.get("url", "")).strip()

    if not raw_url:
        return jsonify({"error": "Enter a URL to scan."}), 400
    if len(raw_url) > MAX_URL_LEN:
        return jsonify({"error": f"URL is longer than {MAX_URL_LEN} characters."}), 400
    if any(ord(c) < 32 or ord(c) == 127 for c in raw_url):
        return jsonify({"error": "URL contains control characters."}), 400

    model_key = resolve_model(payload.get("model"))
    if model_key is None:
        return jsonify({"error": "Unknown model. Use 'svm' or 'lr' (Logistic Regression)."}), 400

    model = get_model(model_key)
    if model is None:
        return jsonify({
            "error": "Model files not found. Run: python generate_dataset.py && python train_models.py"
        }), 503

    scanned_url = normalize_url(raw_url)
    features = extract_features(scanned_url)
    frame = pd.DataFrame([[features[n] for n in FEATURE_NAMES]], columns=FEATURE_NAMES)

    malicious_idx = list(model.classes_).index(1)
    p_malicious = float(model.predict_proba(frame)[0][malicious_idx])
    # Classification, alert level and the displayed score all use the same rounded
    # value, so what the analyst sees is always internally consistent.
    score = round(p_malicious * 100, 1)
    classification = "MALICIOUS" if score >= HIGH_THRESHOLD else "LEGITIMATE"

    host = hostname_of(scanned_url)
    alert = build_alert(score, host, bool(features["has_ip_address"]), scanned_url)
    timestamp = _utc_now().strftime("%Y-%m-%d %H:%M:%S")

    logged = payload.get("log", True) is not False
    if logged:
        add_log({
            "timestamp": timestamp,
            "url": scanned_url,
            "model": MODEL_REGISTRY[model_key]["label"],
            "status": classification,
            "risk_score": score,
            "level": alert["level"],
            "action": alert["log_action"],
        })

    return jsonify({
        "url": raw_url,
        "scanned_url": scanned_url,
        "host": host,
        "model": MODEL_REGISTRY[model_key]["label"],
        "model_key": model_key,
        "classification": classification,
        "risk_score": score,
        "probability_malicious": round(p_malicious, 4),
        "features": features,
        "alert": {**{k: v for k, v in alert.items() if k != "log_action"}, "simulated": True},
        "timestamp": timestamp,
        "logged": logged,
    })


@app.get("/api/logs")
def api_logs():
    return jsonify({"logs": get_logs()})


@app.post("/api/logs/clear")
def api_logs_clear():
    clear_logs()
    return jsonify({"logs": []})


if __name__ == "__main__":
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG") == "1",
    )
