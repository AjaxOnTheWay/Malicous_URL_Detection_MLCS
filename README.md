# Activity 2: Legitimate vs. Malicious URL Classification using ML

A Flask web app with a dark-mode, Splunk-style SOC dashboard. It classifies URLs as **LEGITIMATE** or **MALICIOUS** using Logistic Regression and an RBF SVM, scores risk from 0 to 100, and simulates an automated SOC response. It runs fully offline: no CDN, no external downloads.

## Features

- URL scanner with a model selector (SVM or Logistic Regression) and quick demo buttons
- Risk score (0-100) derived from `predict_proba()`, plus a breakdown of the 6 extracted features
- Simulated SOC response based on the risk score
- Model performance comparison (Accuracy, Precision, Recall, F1-Score, confusion matrices)
- Live SIEM incident log, kept per session, with a Clear Logs button

## Project structure

```
soc-url-classifier/
├── app.py               Flask backend (dashboard + REST API)
├── features.py          Deterministic URL feature extraction
├── generate_dataset.py  Builds data/urldata.csv (500 legitimate + 500 malicious)
├── train_models.py      Trains, evaluates and saves both models
├── smoke_test.py        Automated end-to-end check of the app
├── requirements.txt
├── templates/
│   └── dashboard.html  SOC dashboard (vanilla HTML/CSS/JS)
├── data/urldata.csv     Generated dataset
└── models/              lr_model.pkl, svm_model.pkl, metrics.pkl
```

## Quick start

```bash
# 1. Create a virtual environment and install dependencies
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Build the dataset
python generate_dataset.py

# 3. Train both models (prints the comparison table, saves models/*.pkl)
python train_models.py

# 4. (Optional) Run the smoke test
python smoke_test.py

# 5. Start the dashboard
python app.py
```

Open http://127.0.0.1:5000

Optional environment variables: `HOST`, `PORT`, `SECRET_KEY`, `FLASK_DEBUG=1`.

## Extracted features

| Feature | Description |
|---|---|
| `url_length` | Total characters in the URL |
| `has_https` | 1 if the scheme is https, else 0 |
| `dot_count` | Number of `.` characters |
| `special_char_count` | Count of `@ - _ ? = % &` |
| `suspicious_word_count` | Occurrences of keywords such as login, verify, update, secure, account, banking, free, lucky, bonus, click, urgent, signin, confirm, wallet, paypal, admin |
| `has_ip_address` | 1 if the hostname contains an IPv4 address |

## Models

Both models are scikit-learn pipelines: `StandardScaler` followed by either `LogisticRegression` or `SVC(kernel='rbf', probability=True)`. They are trained on an 80% split and evaluated on a stratified 20% hold-out set (seed 42).

## Risk score and simulated SOC response

| Risk score | Alert | Simulated response |
|---|---|---|
| 80 and above | CRITICAL ALERT | Firewall IP/domain block, DNS sinkhole, SOC incident ticket |
| 50 to 79 | HIGH WARNING | Endpoint sandbox isolation, analyst triage alert |
| Below 50 | INFO / SAFE | Traffic allowed, logged to SIEM baseline |

Actions are simulated only. No real network changes are made.

## REST API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Dashboard |
| POST | `/api/scan` | Classify a URL |
| GET | `/api/metrics` | Model comparison metrics |
| GET | `/api/logs` | Session incident log |
| POST | `/api/logs/clear` | Clear session log |
| GET | `/api/health` | Model availability |

Example:

```bash
curl -X POST http://127.0.0.1:5000/api/scan \
  -H "Content-Type: application/json" \
  -d '{"url": "http://192.168.1.45/secure-login-verify-account?id=99", "model": "svm"}'
```

`model` accepts `svm` or `lr`. Add `"log": false` to skip writing to the session log.

## Using your own dataset

Any CSV with `url` and `label` columns works (labels like `legitimate`/`malicious`, `benign`/`phishing`, `good`/`bad`, or `0`/`1`):

```bash
python train_models.py --data path/to/dataset.csv
# For very large files, subsample so the SVM trains in reasonable time:
python train_models.py --data path/to/dataset.csv --max-samples 20000
```

## Notes and limitations

- The bundled dataset is synthetic. It is realistic but overlapping, so expect roughly 88-90% accuracy rather than 100%.
- Six lexical features cannot catch every malicious URL (for example, keyword-free `https` links).
- Incident logs are stored in server memory per browser session and are cleared on restart.
- URLs typed without a scheme are scanned as `http://`.
- This is a teaching and demo project, not a production security control.
