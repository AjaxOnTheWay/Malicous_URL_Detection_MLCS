"""Deterministic URL feature extraction.

The same URL always produces the same six features, in the same order. Both the
training pipeline (train_models.py) and the live scanner (app.py) import this
module, so there is no train/serve skew.

Feature definitions
-------------------
url_length             Total number of characters in the URL.
has_https              1 if the URL scheme is https, else 0.
dot_count              Number of "." characters in the whole URL.
special_char_count     Combined count of  @  -  _  ?  =  %  &
suspicious_word_count  Total occurrences of phishing/malware keywords
                       (case-insensitive substring match, e.g. "accounts"
                       counts as "account").
has_ip_address         1 if the hostname contains an IPv4 address, else 0.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List
from urllib.parse import urlparse

FEATURE_NAMES: List[str] = [
    "url_length",
    "has_https",
    "dot_count",
    "special_char_count",
    "suspicious_word_count",
    "has_ip_address",
]

SPECIAL_CHARS = ("@", "-", "_", "?", "=", "%", "&")

SUSPICIOUS_WORDS = (
    "login", "verify", "update", "secure", "account", "banking", "free",
    "lucky", "bonus", "click", "urgent", "signin", "confirm", "wallet",
    "paypal", "admin",
)

_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
IPV4_RE = re.compile(rf"(?<![\d.]){_OCTET}(?:\.{_OCTET}){{3}}(?![\d.])")


def _scheme(url: str) -> str:
    try:
        return urlparse(url).scheme.lower()
    except ValueError:
        return ""


def hostname_of(url: str) -> str:
    """Lower-cased hostname (no port, no userinfo); '' if it cannot be parsed."""
    try:
        parsed = urlparse(url if "://" in url else "//" + url)
        return parsed.hostname or ""
    except ValueError:
        return ""


def extract_features(url: str) -> Dict[str, int]:
    """Return an ordered dict of the six URL features."""
    url = (url or "").strip()
    lowered = url.lower()
    host = hostname_of(url)

    return {
        "url_length": len(url),
        "has_https": int(_scheme(url) == "https"),
        "dot_count": url.count("."),
        "special_char_count": sum(url.count(ch) for ch in SPECIAL_CHARS),
        "suspicious_word_count": sum(lowered.count(w) for w in SUSPICIOUS_WORDS),
        "has_ip_address": int(bool(IPV4_RE.search(host))),
    }


def feature_vector(url: str) -> List[int]:
    """Feature values in FEATURE_NAMES order."""
    feats = extract_features(url)
    return [feats[name] for name in FEATURE_NAMES]


def features_dataframe(urls: Iterable[str]):
    """Build a pandas DataFrame (columns = FEATURE_NAMES) for many URLs."""
    import pandas as pd

    return pd.DataFrame([feature_vector(u) for u in urls], columns=FEATURE_NAMES)


if __name__ == "__main__":
    for sample in ("https://www.github.com", "http://192.168.1.45/secure-login-verify-account?id=99"):
        print(sample)
        for k, v in extract_features(sample).items():
            print(f"  {k:<22} {v}")
