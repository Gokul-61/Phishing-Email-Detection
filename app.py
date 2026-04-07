"""
🛡️ Phishing Email Detector — Flask Web App
Run: python app.py
Then open: http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.sparse import hstack, csr_matrix
import os



nltk.download('stopwords')

app = Flask(__name__)

# ── Load saved models ──────────────────────────────────────────────────────
model  = joblib.load('best_phishing_model.pkl')
tfidf  = joblib.load('tfidf_vectorizer.pkl')

stop_words = set(stopwords.words('english'))
stemmer    = PorterStemmer()

PHISHING_KEYWORDS = [
    'urgent', 'verify', 'login', 'suspended', 'click here',
    'click the link', 'paypal', 'prize', 'winner', 'congratulations',
    'limited time', 'expire', 'unusual activity', 'restore access',
    'permanent closure', 'permanently locked', 'immediately',
    'act now', 'do not share', 'send your details',
    'bank details', 'id proof', 'personal details',
    'lottery', 'selected', 'lucky winner', 'claim your'
]

URL_DEPENDENT_KEYWORDS = [
    'verify', 'confirm', 'update your', 'restore', 'reactivate',
    'validate', 'click below', 'click here'
]

PHISHING_SUBJECT_WORDS = [
    'urgent', 'suspended', 'compromised', 'unauthorized',
    'verify now', 'action required', 'your account', 'congratulations',
    'you have won', 'lottery', 'job offer', 'delivery failed'
]

SAFE_CONTEXT_WORDS = [
    'timetable', 'schedule', 'uploaded', 'portal', 'reminder',
    'meeting', 'agenda', 'newsletter', 'weekly', 'order confirmed',
    'order #', 'track your order', 'password was changed',
    'successfully updated', 'successfully placed', 'shopping',
    'exam', 'semester', 'academic', 'lecture', 'result'
]

FEATURE_NAMES = [
    'num_links', 'num_suspicious_words', 'email_length',
    'special_chars', 'num_digits', 'upper_ratio',
    'exclamation_count', 'suspicious_url_score'
]

# ── Helper functions ───────────────────────────────────────────────────────

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words and len(w) > 2]
    return " ".join(words)

def extract_features(text):
    text = str(text)
    text_lower = text.lower()

    subject = ""
    body = text_lower
    if "subject:" in text_lower:
        parts = text_lower.split("\n", 1)
        subject = parts[0].replace("subject:", "").strip()
        body = parts[1] if len(parts) > 1 else text_lower

    num_links = len(re.findall(r'http[s]?://|www\.', text_lower))

    direct_matches  = sum(1 for kw in PHISHING_KEYWORDS if kw in text_lower)
    url_dep_matches = sum(1 for kw in URL_DEPENDENT_KEYWORDS if kw in text_lower) if num_links > 0 else 0
    subject_matches = sum(1 for kw in PHISHING_SUBJECT_WORDS if kw in subject)
    safe_penalty    = sum(1 for kw in SAFE_CONTEXT_WORDS if kw in text_lower)
    num_suspicious_words = max(0, direct_matches + url_dep_matches + subject_matches - safe_penalty)

    email_length  = len(text)
    special_chars = len(re.findall(r'[@!\$#%]', text))
    num_digits    = len(re.findall(r'\d', text))
    if re.search(r'order\s*#\s*\d+|order\s+number\s+\d+', text_lower):
        num_digits = max(0, num_digits - 5)

    alpha_chars     = [c for c in text if c.isalpha()]
    upper_ratio     = sum(1 for c in alpha_chars if c.isupper()) / (len(alpha_chars) + 1)
    exclamation_count = text.count('!')

    suspicious_url_score = 0
    urls = re.findall(r'https?://([^\s/]+)', text_lower)
    for domain in urls:
        if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', domain):
            suspicious_url_score += 3
        if any(kw in domain for kw in ['secure','login','verify','paypal','bank','account']):
            suspicious_url_score += 2
        if any(domain.endswith(ext) for ext in ['.xyz','.net','.info','.tk','.ml']):
            suspicious_url_score += 1
        if domain.count('-') >= 2:
            suspicious_url_score += 1

    return [num_links, num_suspicious_words, email_length,
            special_chars, num_digits, upper_ratio,
            exclamation_count, suspicious_url_score]

def extract_urls(text):
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+|www\.[^\s<>"{}|\\^`\[\]]+'
    return re.findall(url_pattern, str(text))

def is_url_suspicious(url):
    flags = []
    if len(url) > 75:
        flags.append("URL is too long (>75 chars)")
    if re.search(r'https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url):
        flags.append("Uses IP address instead of domain name")
    domain_part = re.sub(r'https?://', '', url).split('/')[0]
    if domain_part.count('.') > 3:
        flags.append("Too many subdomains (possible spoofing)")
    if '@' in url:
        flags.append("Contains '@' symbol (redirect trick)")
    if re.search(r'\d{4,}', domain_part):
        flags.append("Domain contains suspicious numbers")
    suspicious_url_words = ['login','verify','secure','account','update','confirm','paypal','bank']
    found = [w for w in suspicious_url_words if w in url.lower()]
    if found:
        flags.append(f"Suspicious keywords: {', '.join(found)}")
    shorteners = ['bit.ly','tinyurl','goo.gl','t.co','ow.ly','buff.ly','short.link']
    if any(s in url.lower() for s in shorteners):
        flags.append("Uses URL shortener (hides real destination)")
    return flags

def analyze_urls(text):
    urls = extract_urls(text)
    if not urls:
        return {"total": 0, "suspicious": [], "safe": [], "risk": "LOW"}
    suspicious, safe = [], []
    for url in urls:
        flags = is_url_suspicious(url)
        if flags:
            suspicious.append({"url": url[:80], "reasons": flags})
        else:
            safe.append(url[:80])
    ratio = len(suspicious) / len(urls)
    risk  = "HIGH" if ratio > 0.5 else "MEDIUM" if ratio > 0 else "LOW"
    return {"total": len(urls), "suspicious": suspicious, "safe": safe, "risk": risk}

def get_trigger_words(email_text, top_n=8):
    cleaned   = preprocess_text(email_text)
    text_vec  = tfidf.transform([cleaned])
    feat_names= tfidf.get_feature_names_out()
    # Use model coef if LR, else return keyword matches
    try:
        coefs = model.coef_[0][:len(feat_names)]
        text_arr    = text_vec.toarray()[0]
        word_scores = [(feat_names[i], float(coefs[i]))
                       for i in range(len(feat_names))
                       if text_arr[i] > 0 and coefs[i] > 0]
        word_scores.sort(key=lambda x: x[1], reverse=True)
        return [w for w, _ in word_scores[:top_n]]
    except Exception:
        return [w for w in PHISHING_KEYWORDS if w in email_text.lower()][:top_n]

def predict(email_text):
    cleaned   = preprocess_text(email_text)
    text_vec  = tfidf.transform([cleaned])
    eng_feats = np.array(extract_features(email_text)).reshape(1, -1)

    try:
        combined = hstack([text_vec, csr_matrix(eng_feats)])
        pred     = model.predict(combined)[0]
        try:
            proba      = model.predict_proba(combined)[0]
            confidence = round(float(max(proba)) * 100, 1)
        except Exception:
            confidence = None
    except Exception:
        pred = model.predict(text_vec)[0]
        confidence = None

    url_info = analyze_urls(email_text)
    triggers = get_trigger_words(email_text) if pred == 1 else []
    feat_dict = dict(zip(FEATURE_NAMES, [round(float(v), 3) for v in eng_feats[0]]))

    return {
        "label":       int(pred),
        "verdict":     "PHISHING" if pred == 1 else "SAFE",
        "confidence":  confidence,
        "triggers":    triggers,
        "url_analysis":url_info,
        "features":    feat_dict
    }

# ── HTML Template ──────────────────────────────────────────────────────────



# ── Routes ─────────────────────────────────────────────────────────────────

from flask import render_template

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    data  = request.get_json()
    email = data.get('email', '')
    if not email.strip():
        return jsonify({"error": "Empty email"}), 400
    result = predict(email)
    return jsonify(result)

if __name__ == '__main__':
    print("🛡️  Phishing Detector starting on http://localhost:5000")
    print("   Make sure best_phishing_model.pkl and tfidf_vectorizer.pkl are in the same folder.")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
