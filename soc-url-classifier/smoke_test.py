#!/usr/bin/env python3
"""Smoke test for app.py using Flask's test client (no server needed).

Run:  python smoke_test.py      (exit code 0 = all checks passed)
"""
from __future__ import annotations

import sys

from app import app

BENIGN = "https://www.github.com"
MALICIOUS = "http://192.168.1.45/secure-login-verify-account?id=99"

failures = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(name)


def main() -> int:
    with app.test_client() as client:
        print("Pages and metrics")
        check("GET / serves the dashboard", client.get("/").status_code == 200)
        health = client.get("/api/health").get_json()
        check("both models loaded", health["ready"], str(health["models_loaded"]))
        metrics = client.get("/api/metrics").get_json()
        check("metrics.pkl served", set(metrics["models"]) == {"Logistic Regression", "SVM"})

        for model in ("svm", "lr"):
            print(f"\nScans with model={model}")
            good = client.post("/api/scan", json={"url": BENIGN, "model": model}).get_json()
            bad = client.post("/api/scan", json={"url": MALICIOUS, "model": model}).get_json()
            print(f"  benign    {good['classification']:<10} risk={good['risk_score']:>5}  {good['alert']['label']}")
            print(f"  malicious {bad['classification']:<10} risk={bad['risk_score']:>5}  {bad['alert']['label']}")

            check("benign URL classified LEGITIMATE", good["classification"] == "LEGITIMATE")
            check("benign risk score below 50", good["risk_score"] < 50, str(good["risk_score"]))
            check("benign -> INFO / SAFE alert", good["alert"]["level"] == "INFO")
            check("malicious URL classified MALICIOUS", bad["classification"] == "MALICIOUS")
            check("malicious risk score >= 80", bad["risk_score"] >= 80, str(bad["risk_score"]))
            check("malicious -> CRITICAL ALERT", bad["alert"]["level"] == "CRITICAL")
            titles = [a["title"] for a in bad["alert"]["actions"]]
            check("critical response has block + sinkhole + ticket",
                  titles == ["Firewall block", "DNS sinkhole", "SOC incident ticket"], str(titles))
            check("malicious feature breakdown", bad["features"] == {
                "url_length": len(MALICIOUS), "has_https": 0, "dot_count": 3,
                "special_char_count": 5, "suspicious_word_count": 4, "has_ip_address": 1})
            check("benign feature breakdown", good["features"] == {
                "url_length": len(BENIGN), "has_https": 1, "dot_count": 2,
                "special_char_count": 0, "suspicious_word_count": 0, "has_ip_address": 0})

        print("\nSession SIEM log")
        logs = client.get("/api/logs").get_json()["logs"]
        check("4 scans logged in session", len(logs) == 4, f"got {len(logs)}")
        check("log entry has all columns", {"timestamp", "url", "model", "status", "risk_score", "action"} <= set(logs[0]))
        check("clear logs empties the table", client.post("/api/logs/clear").get_json()["logs"] == [])
        check("log stays empty afterwards", client.get("/api/logs").get_json()["logs"] == [])

        print("\nValidation")
        check("empty URL -> 400", client.post("/api/scan", json={"url": ""}).status_code == 400)
        check("bad model -> 400", client.post("/api/scan", json={"url": BENIGN, "model": "xgb"}).status_code == 400)
        check("non-JSON body -> 400", client.post("/api/scan", data="url=x").status_code == 400)
        bare = client.post("/api/scan", json={"url": "github.com", "log": False}).get_json()
        check("bare host is normalised and not logged", bare["scanned_url"] == "http://github.com" and not bare["logged"])

    print(f"\n{'ALL CHECKS PASSED' if not failures else str(len(failures)) + ' CHECK(S) FAILED: ' + ', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
