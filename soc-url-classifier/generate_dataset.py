#!/usr/bin/env python3
"""Generate a balanced, offline URL dataset (data/urldata.csv).

Output columns: url,label   (label is "legitimate" or "malicious")

The data is synthetic but deliberately realistic and *not* trivially separable:
  * some legitimate URLs contain words like "login", "account" or "update",
    long tracking query strings, hyphens and underscores;
  * some malicious URLs use https, or carry no suspicious keyword at all.
That overlap keeps model metrics realistic (below 100%) instead of perfect.

Any standard Kaggle CSV with `url` and `label` columns can be used instead:
    python train_models.py --data path/to/kaggle.csv
"""
from __future__ import annotations

import argparse
import csv
import random
import string
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# Legitimate URL building blocks
# --------------------------------------------------------------------------- #
LEGIT_SITES = [
    "google.com", "github.com", "wikipedia.org", "stackoverflow.com", "microsoft.com",
    "apple.com", "amazon.com", "nytimes.com", "bbc.co.uk", "cnn.com", "reddit.com",
    "linkedin.com", "youtube.com", "netflix.com", "spotify.com", "mozilla.org",
    "python.org", "kaggle.com", "medium.com", "dropbox.com", "zoom.us", "slack.com",
    "adobe.com", "ibm.com", "oracle.com", "cloudflare.com", "nasa.gov", "harvard.edu",
    "mit.edu", "iitp.ac.in", "nptel.ac.in", "coursera.org", "udemy.com", "flipkart.com",
    "irctc.co.in", "thehindu.com", "bbc.com", "theguardian.com", "stripe.com",
    "paypal.com", "chase.com", "bankofamerica.com", "walmart.com", "ebay.com",
    "etsy.com", "twitch.tv", "pinterest.com", "quora.com", "khanacademy.org",
    "w3schools.com", "geeksforgeeks.org", "nvidia.com", "intel.com", "docker.com",
    "kubernetes.io", "scikit-learn.org", "pandas.pydata.org", "numpy.org",
    "anthropic.com", "openai.com",
]
LEGIT_SUBS = ["www"] * 6 + [""] * 2 + [
    "docs", "blog", "support", "news", "mail", "help", "developer", "learn", "store", "en",
]
SECTIONS = [
    "about", "products", "docs", "blog", "news", "careers", "pricing", "contact",
    "research", "downloads", "courses", "community", "help", "search", "category",
    "topics", "wiki", "questions", "articles", "press", "events", "learn", "tutorials",
]
SLUG_WORDS = [
    "how", "to", "learn", "python", "machine", "learning", "guide", "best", "practices",
    "intro", "tutorial", "getting", "started", "release", "notes", "annual", "report",
    "data", "science", "cloud", "security", "network", "design", "review", "tips",
    "beginners", "overview", "roadmap", "privacy", "policy", "faq", "history",
]
LEGIT_SENSITIVE_PATHS = [
    "login", "signin", "account/settings", "accounts/security", "admin/dashboard",
    "update-your-profile", "secure/checkout", "confirm-subscription", "free-trial",
    "wallet/overview", "banking/overview", "account/verify-email", "login/help",
]
QUERY_BUILDERS = [
    lambda r: "?q=" + "+".join(r.sample(SLUG_WORDS, 2)),
    lambda r: f"?id={r.randint(1000, 99999999)}",
    lambda r: f"?v={token(r, 11)}&list={token(r, 12)}",
    lambda r: "?utm_source=newsletter&utm_medium=email&utm_campaign=" + "_".join(r.sample(SLUG_WORDS, 2)),
    lambda r: f"?page={r.randint(1, 50)}&sort={r.choice(['new', 'popular', 'top'])}",
    lambda r: f"?ref={token(r, 8)}%2F{token(r, 4)}&lang=en",
]

# --------------------------------------------------------------------------- #
# Malicious URL building blocks
# --------------------------------------------------------------------------- #
BRANDS = [
    "paypal", "amazon", "apple", "microsoft", "netflix", "google", "facebook",
    "instagram", "hdfc", "sbi", "icici", "chase", "wellsfargo", "dhl", "fedex",
    "irs", "coinbase", "metamask", "binance", "dropbox", "outlook", "whatsapp",
]
TLDS = ["xyz", "top", "tk", "ml", "ga", "cf", "gq", "info", "click", "online",
        "site", "ru", "cn", "biz", "live", "icu", "work", "club", "shop"]
MAL_WORDS = [
    "login", "verify", "update", "secure", "account", "banking", "free", "lucky",
    "bonus", "click", "urgent", "signin", "confirm", "wallet", "admin", "password",
    "auth", "billing", "recover", "support", "alert", "session", "validate",
]
PRIZE_WORDS = ["free", "lucky", "bonus", "winner", "prize", "gift", "claim",
               "iphone", "reward", "cash", "click", "jackpot"]


def token(rng: random.Random, n: int, alphabet: str = string.ascii_letters + string.digits) -> str:
    return "".join(rng.choice(alphabet) for _ in range(n))


# --------------------------------------------------------------------------- #
# Generators
# --------------------------------------------------------------------------- #
LEGIT_INTERNAL = [
    "http://192.168.0.1/admin", "http://192.168.1.1/login", "http://10.0.0.1/admin/status",
    "http://192.168.1.254/index.html", "http://172.16.0.1/dashboard", "http://192.168.0.100:8080/admin",
]


def legit_url(rng: random.Random) -> str:
    if rng.random() < 0.03:                           # router / intranet pages
        base = rng.choice(LEGIT_INTERNAL)
        return base if rng.random() < 0.5 else f"{base}?lang=en&page={rng.randint(1, 9)}"
    site = rng.choice(LEGIT_SITES)
    scheme = "https" if rng.random() < 0.9 else "http"
    sub = rng.choice(LEGIT_SUBS)
    host = f"{sub}.{site}" if sub else site
    roll = rng.random()
    path, force_query = "", False

    if roll < 0.12:                                   # bare homepage
        path = ""
    elif roll < 0.24:                                 # legitimate "sensitive" pages
        path = "/" + rng.choice(LEGIT_SENSITIVE_PATHS)
    elif roll < 0.55:                                 # section pages
        path = "/" + "/".join(rng.sample(SECTIONS, rng.randint(1, 2)))
    elif roll < 0.80:                                 # dated article, hyphenated slug
        slug = "-".join(rng.sample(SLUG_WORDS, rng.randint(2, 5)))
        path = f"/{rng.randint(2015, 2025)}/{rng.randint(1, 12):02d}/{slug}"
    else:                                             # long query strings
        path = "/" + rng.choice(SECTIONS)
        force_query = True

    query = ""
    if force_query or rng.random() < 0.08:
        query = rng.choice(QUERY_BUILDERS)(rng)
    return f"{scheme}://{host}{path}{query}"


def random_ip(rng: random.Random) -> str:
    kind = rng.random()
    if kind < 0.25:
        return f"192.168.{rng.randint(0, 255)}.{rng.randint(1, 254)}"
    if kind < 0.35:
        return f"10.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"
    return ".".join(str(rng.randint(1, 254)) for _ in range(4))


def mal_ip(rng: random.Random) -> str:
    port = rng.choice(["", "", "", ":8080", ":8888", ":8000", ":81"])
    words = rng.sample(MAL_WORDS, rng.randint(1, 4))
    path = "/" + "-".join(words) + rng.choice(["", ".php", ".html", "/index.php"])
    query = rng.choice(["", "", f"?id={rng.randint(1, 999)}",
                        f"?session={token(rng, 16)}&user={token(rng, 6)}"])
    return f"http://{random_ip(rng)}{port}{path}{query}"


def mal_lookalike(rng: random.Random) -> str:
    parts = [rng.choice(BRANDS)] + rng.sample(MAL_WORDS, rng.randint(1, 2))
    rng.shuffle(parts)
    host = "-".join(parts) + "." + rng.choice(TLDS)
    scheme = "https" if rng.random() < 0.3 else "http"
    path = rng.choice([
        "", "/", "/" + "/".join(rng.sample(MAL_WORDS, 2)),
        "/" + rng.choice(MAL_WORDS) + ".php", "/wp-content/" + rng.choice(MAL_WORDS) + ".html",
    ])
    query = rng.choice(["", "", f"?id={rng.randint(1, 99999)}",
                        f"?token={token(rng, 20)}&email=user%40mail.com"])
    return f"{scheme}://{host}{path}{query}"


def mal_stacked(rng: random.Random) -> str:
    brand = rng.choice(BRANDS)
    prefix = rng.choice(["", "www.", "secure.", "my.", "login."])
    host = f"{prefix}{brand}.com.{rng.choice(MAL_WORDS)}-{rng.choice(MAL_WORDS)}.{rng.choice(TLDS)}"
    scheme = "https" if rng.random() < 0.35 else "http"
    path = "/" + rng.choice(["signin", "login", "verify", "index.php", "webscr", "confirm"])
    return f"{scheme}://{host}{path}"


def mal_prize(rng: random.Random) -> str:
    host = "-".join(rng.sample(PRIZE_WORDS, 2)) + str(rng.randint(1, 999)) + "." + rng.choice(TLDS)
    path = rng.choice(["/", "/click", "/claim.php", "/win/" + token(rng, 5, string.ascii_lowercase)])
    query = f"?ref={token(rng, 8)}&click={token(rng, 6)}&bonus=1"
    scheme = "https" if rng.random() < 0.2 else "http"
    return f"{scheme}://{host}{path}{query}"


def mal_at_sign(rng: random.Random) -> str:
    decoy = rng.choice(LEGIT_SITES)
    host = token(rng, rng.randint(6, 10), string.ascii_lowercase) + "." + rng.choice(TLDS)
    return f"http://www.{decoy}@{host}/{rng.choice(MAL_WORDS)}"


def mal_stealth(rng: random.Random) -> str:
    """Malware-style URL with no obvious keyword (hard negatives for the model)."""
    lower = string.ascii_lowercase
    host = token(rng, rng.randint(5, 10), lower)
    if rng.random() < 0.55:
        host += "-" + token(rng, rng.randint(2, 5), lower + string.digits)
    host += str(rng.randint(1, 9999)) + "." + rng.choice(TLDS)
    scheme = "https" if rng.random() < 0.35 else "http"
    segs = "/".join(token(rng, rng.randint(3, 8), lower + string.digits + "_")
                    for _ in range(rng.randint(1, 3)))
    ext = rng.choice([".exe", ".zip", ".php", ".js", "", ".bin"])
    query = rng.choice([
        "", "",
        f"?a={token(rng, 6)}&b={token(rng, 6)}",
        f"?x={token(rng, 10)}",
        f"?id={rng.randint(1, 9999)}&k={token(rng, 8)}&t={rng.randint(10**9, 10**10)}&r={token(rng, 5)}",
        f"?d={token(rng, 12)}%3D&s={token(rng, 6)}%26v={rng.randint(1, 99)}",
    ])
    return f"{scheme}://{host}/{segs}{ext}{query}"


MALICIOUS_MIX = [
    (mal_ip, 0.18), (mal_lookalike, 0.32), (mal_stacked, 0.14),
    (mal_prize, 0.12), (mal_at_sign, 0.06), (mal_stealth, 0.12),
]


def malicious_url(rng: random.Random) -> str:
    funcs, weights = zip(*MALICIOUS_MIX)
    return rng.choices(funcs, weights=weights, k=1)[0](rng)


def build_class(rng: random.Random, n: int, fn, seen: set) -> list:
    urls, attempts = [], 0
    while len(urls) < n:
        attempts += 1
        if attempts > n * 200:
            raise RuntimeError("Could not generate enough unique URLs; lower --per-class.")
        url = fn(rng)
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default=str(BASE_DIR / "data" / "urldata.csv"))
    ap.add_argument("--per-class", type=int, default=500, help="samples per class (default 500)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    seen: set = set()
    rows = [(u, "legitimate") for u in build_class(rng, args.per_class, legit_url, seen)]
    rows += [(u, "malicious") for u in build_class(rng, args.per_class, malicious_url, seen)]
    rng.shuffle(rows)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["url", "label"])
        writer.writerows(rows)

    print(f"[dataset] wrote {len(rows)} rows to {out}")
    print(f"[dataset] legitimate={args.per_class}  malicious={args.per_class}  seed={args.seed}")
    print("[dataset] sample rows:")
    for url, label in rows[:4]:
        print(f"    {label:<10} {url}")


if __name__ == "__main__":
    main()
