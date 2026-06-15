"""
============================================================
  Intrusion Detection System (IDS) with Machine Learning
  Internship Project 3
  Author : KUNAL SINGH
  Tools  : Python, Scikit-learn, NumPy, Pandas, Matplotlib
  Dataset: Synthetic network traffic (KDD-Cup inspired features)
  Models : Random Forest, Isolation Forest (anomaly), SVM
============================================================
"""
"""
PROJECT LIMITATION:

This project uses synthetic KDD-inspired network traffic data for
educational demonstration. The reported accuracy should not be
interpreted as real-world IDS performance.

A production-level version should be evaluated using real datasets
such as NSL-KDD, CIC-IDS2017, or network flow data extracted from PCAP files.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score
)
import warnings, time
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
#  SECTION 1 — Synthetic Dataset Generation
#  (Mimics the KDD Cup 99 / NSL-KDD feature space)
# ─────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment",
    "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells",
    "num_access_files", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "rerror_rate",
    "same_srv_rate", "diff_srv_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate",
]

ATTACK_TYPES = {
    # label: (weight, description)
    "normal"   : (0.50, "Legitimate traffic"),
    "dos"      : (0.20, "Denial of Service (SYN flood, Ping of Death)"),
    "probe"    : (0.12, "Port/network scanning (Nmap, Nessus)"),
    "r2l"      : (0.10, "Remote-to-Local exploit (brute-force SSH)"),
    "u2r"      : (0.08, "User-to-Root privilege escalation"),
}

np.random.seed(42)


def _gen_normal(n: int) -> np.ndarray:
    return np.column_stack([
        np.random.exponential(5,  n),          # duration
        np.random.randint(0, 3,   n),          # protocol_type
        np.random.randint(0, 20,  n),          # service
        np.random.randint(0, 11,  n),          # flag
        np.random.exponential(1500, n),        # src_bytes
        np.random.exponential(2000, n),        # dst_bytes
        np.zeros(n),                           # land
        np.random.poisson(0.1, n),             # wrong_fragment
        np.zeros(n),                           # urgent
        np.random.poisson(2,   n),             # hot
        np.zeros(n),                           # num_failed_logins
        np.ones(n),                            # logged_in
        np.zeros(n),                           # num_compromised
        np.zeros(n),                           # root_shell
        np.zeros(n),                           # su_attempted
        np.zeros(n),                           # num_root
        np.random.poisson(0.5, n),             # num_file_creations
        np.zeros(n),                           # num_shells
        np.random.poisson(0.5, n),             # num_access_files
        np.zeros(n),                           # is_host_login
        np.zeros(n),                           # is_guest_login
        np.random.randint(1, 50, n),           # count
        np.random.randint(1, 50, n),           # srv_count
        np.random.beta(1, 9,   n),             # serror_rate  (low)
        np.random.beta(1, 9,   n),             # rerror_rate  (low)
        np.random.beta(8, 2,   n),             # same_srv_rate (high)
        np.random.beta(1, 9,   n),             # diff_srv_rate (low)
        np.random.randint(1, 255, n),          # dst_host_count
        np.random.randint(1, 255, n),          # dst_host_srv_count
        np.random.beta(7, 3,   n),             # dst_host_same_srv_rate
    ])


def _gen_dos(n: int) -> np.ndarray:
    """High volume, many connections to same service, SYN errors."""
    base = _gen_normal(n)
    base[:, 4]  = np.random.exponential(50000, n)    # large src_bytes
    base[:, 5]  = np.zeros(n)                         # dst_bytes ~ 0
    base[:, 21] = np.random.randint(200, 512, n)      # count very high
    base[:, 22] = np.random.randint(200, 512, n)      # srv_count high
    base[:, 23] = np.random.beta(9, 1, n)             # serror_rate high
    base[:, 25] = np.random.beta(9, 1, n)             # same_srv_rate high
    return base


def _gen_probe(n: int) -> np.ndarray:
    """Many different services, short durations."""
    base = _gen_normal(n)
    base[:, 0]  = np.random.exponential(0.5, n)       # short duration
    base[:, 4]  = np.random.exponential(200, n)        # small bytes
    base[:, 5]  = np.random.exponential(200, n)
    base[:, 11] = np.zeros(n)                           # not logged in
    base[:, 26] = np.random.beta(8, 2, n)              # diff_srv_rate high
    base[:, 21] = np.random.randint(50, 200, n)        # moderate count
    return base


def _gen_r2l(n: int) -> np.ndarray:
    """Failed logins, guest access."""
    base = _gen_normal(n)
    base[:, 10] = np.random.randint(3, 15, n)         # num_failed_logins
    base[:, 11] = np.zeros(n)                          # not logged in
    base[:, 19] = np.random.randint(0, 2, n)          # is_host_login
    base[:, 20] = np.random.randint(0, 2, n)          # is_guest_login
    base[:, 4]  = np.random.exponential(300, n)
    return base


def _gen_u2r(n: int) -> np.ndarray:
    """Root shell, su attempts, file creation."""
    base = _gen_normal(n)
    base[:, 13] = np.random.randint(0, 2, n)          # root_shell
    base[:, 14] = np.random.randint(0, 2, n)          # su_attempted
    base[:, 15] = np.random.poisson(3, n)             # num_root
    base[:, 17] = np.random.poisson(2, n)             # num_shells
    base[:, 9]  = np.random.randint(5, 20, n)         # hot
    return base


_GENERATORS = {
    "normal": _gen_normal,
    "dos"   : _gen_dos,
    "probe" : _gen_probe,
    "r2l"   : _gen_r2l,
    "u2r"   : _gen_u2r,
}


def generate_dataset(total_samples: int = 20_000) -> pd.DataFrame:
    frames = []
    for label, (weight, _) in ATTACK_TYPES.items():
        n = int(total_samples * weight)
        data  = _GENERATORS[label](n)
        df    = pd.DataFrame(data, columns=FEATURE_NAMES)
        df["label"] = label
        frames.append(df)
    full = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=42)
    return full


# ─────────────────────────────────────────────────────────
#  SECTION 2 — IDS Model
# ─────────────────────────────────────────────────────────

class IntrusionDetectionSystem:
    """
    Three detection layers:
      1. Supervised   – Random Forest (multi-class attack classification)
      2. Supervised   – SVM (binary: normal vs attack)
      3. Unsupervised – Isolation Forest (anomaly / zero-day detection)
    """

    def __init__(self):
        self.scaler    = StandardScaler()
        self.le        = LabelEncoder()
        self.rf        = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        self.svm       = SVC(kernel="rbf", C=1.0, probability=True, random_state=42)
        self.iso_forest = IsolationForest(n_estimators=100, 
                                          contamination=0.05,
                                          random_state=42)
        self.trained   = False

    # ── Training ──────────────────────────────────────────
    def train(self, df: pd.DataFrame):
        X = df[FEATURE_NAMES].values
        y_multi  = self.le.fit_transform(df["label"])
        y_binary = (df["label"] != "normal").astype(int).values

        
        X_tr, X_te, ym_tr, ym_te, yb_tr, yb_te = train_test_split(
            X, y_multi, y_binary,
            test_size=0.2,
            random_state=42,
            stratify=y_multi
        )

        X_tr = self.scaler.fit_transform(X_tr)
        X_te = self.scaler.transform(X_te)
    
        

        print("\n  [1/3] Training Random Forest (multi-class) …", end="", flush=True)
        t0 = time.time()
        self.rf.fit(X_tr, ym_tr)
        print(f" {time.time()-t0:.2f}s")

        print("  [2/3] Training SVM (binary: normal vs attack) …", end="", flush=True)
        t0 = time.time()
        self.svm.fit(X_tr, yb_tr)
        print(f" {time.time()-t0:.2f}s")

        print("  [3/3] Training Isolation Forest (anomaly) …", end="", flush=True)
        t0 = time.time()
        normal_label = self.le.transform(["normal"])[0]
        normal_training_data = X_tr[ym_tr == normal_label]

        self.iso_forest.fit(normal_training_data)
        print(f" {time.time()-t0:.2f}s")

        self.trained = True

        # Evaluate
        self._evaluate(X_te, ym_te, yb_te)
        return self

    def _evaluate(self, X_te, ym_te, yb_te):
        print("\n" + "─"*60)
        print("  EVALUATION RESULTS")
        print("─"*60)

        # Random Forest
        rf_pred = self.rf.predict(X_te)
        print(f"\n▶ Random Forest — Multi-class Accuracy: "
              f"{accuracy_score(ym_te, rf_pred)*100:.2f}%")
        labels = self.le.classes_
        print(classification_report(ym_te, rf_pred,
                                    target_names=labels, digits=3))

        # Feature importance
        fi = pd.Series(self.rf.feature_importances_, index=FEATURE_NAMES)
        top5 = fi.nlargest(5)
        print("  Top-5 most important features (Random Forest):")
        for feat, imp in top5.items():
            bar = "█" * int(imp * 200)
            print(f"    {feat:<28} {imp:.4f}  {bar}")

        # SVM
        svm_pred = self.svm.predict(X_te)
        svm_prob = self.svm.predict_proba(X_te)[:, 1]
        print(f"\n▶ SVM — Binary Accuracy : {accuracy_score(yb_te, svm_pred)*100:.2f}%")
        print(f"  ROC-AUC              : {roc_auc_score(yb_te, svm_prob):.4f}")

        # Isolation Forest
        iso_pred = self.iso_forest.predict(X_te)
        iso_bin  = (iso_pred == -1).astype(int)   # -1 = anomaly
        iso_acc  = accuracy_score(yb_te, iso_bin)
        print(f"\n▶ Isolation Forest — Anomaly Detection Accuracy: {iso_acc*100:.2f}%")

    # ── Live Prediction ───────────────────────────────────
    def predict(self, traffic_features: dict) -> dict:
        """Classify a single network connection."""
        if not self.trained:
            raise RuntimeError("Model not trained yet.")
        vec     = np.array([[traffic_features.get(f, 0) for f in FEATURE_NAMES]])
        vec_sc  = self.scaler.transform(vec)

        rf_class  = self.le.inverse_transform(self.rf.predict(vec_sc))[0]
        rf_proba  = self.rf.predict_proba(vec_sc)[0]
        svm_bin   = int(self.svm.predict(vec_sc)[0])
        iso_flag  = self.iso_forest.predict(vec_sc)[0]

        proba_dict = {cls: float(p)
                      for cls, p in zip(self.le.classes_, rf_proba)}
        confidence = max(rf_proba)

        return {
            "predicted_attack_type": rf_class,
            "is_attack"            : rf_class != "normal",
            "confidence"           : round(confidence, 4),
            "class_probabilities"  : {k: round(v, 4) for k, v in proba_dict.items()},
            "svm_attack_flag"      : bool(svm_bin),
            "anomaly_detected"     : iso_flag == -1,
        }


# ─────────────────────────────────────────────────────────
#  SECTION 3 — Alert Engine
# ─────────────────────────────────────────────────────────

SEVERITY = {
    "normal": "INFO",
    "probe" : "LOW",
    "r2l"   : "MEDIUM",
    "dos"   : "HIGH",
    "u2r"   : "CRITICAL",
}

def raise_alert(result: dict, src_ip: str = "??.??.??.??"):
    severity = SEVERITY.get(result["predicted_attack_type"], "UNKNOWN")
    status   = "🚨 ALERT" if result["is_attack"] else "✅ CLEAN"
    print(f"""
  ┌────────────────────────────────────────────────────┐
  │  IDS ALERT ENGINE                                  │
  ├────────────────────────────────────────────────────┤
  │  Source IP      : {src_ip:<32}│
  │  Status         : {status:<32}│
  │  Attack Type    : {result['predicted_attack_type']:<32}│
  │  Severity       : {severity:<32}│
  │  Confidence     : {result['confidence']:<32}│
  │  SVM Flag       : {str(result['svm_attack_flag']):<32}│
  │  Anomaly Flag   : {str(result['anomaly_detected']):<32}│
  └────────────────────────────────────────────────────┘""")


# ─────────────────────────────────────────────────────────
#  DEMO
# ─────────────────────────────────────────────────────────

def run_demo():
    print("\n" + "█"*60)
    print("   INTRUSION DETECTION SYSTEM — ML-BASED  (PROJECT 3)")
    print("█"*60)

    print("\n[*] Generating synthetic network traffic dataset …")
    df = generate_dataset(total_samples=20_000)
    print(f"    Samples : {len(df):,}")
    print(f"    Features: {len(FEATURE_NAMES)}")
    print("    Class distribution:")
    for label, cnt in df["label"].value_counts().items():
        desc = ATTACK_TYPES[label][1]
        pct  = cnt / len(df) * 100
        bar  = "█" * int(pct / 2)
        print(f"      {label:<10} {cnt:>5} ({pct:4.1f}%)  {bar}  {desc}")

    print("\n[*] Training IDS models …")
    ids = IntrusionDetectionSystem()
    ids.train(df)

    # ── Simulate live traffic ──────────────────────────────
    print("\n" + "═"*60)
    print("  LIVE TRAFFIC SIMULATION")
    print("═"*60)

    test_cases = [
        ("192.168.1.10", {f: 0 for f in FEATURE_NAMES} | {
            "duration": 8, "src_bytes": 1800, "dst_bytes": 2200,
            "logged_in": 1, "count": 10, "same_srv_rate": 0.9,
            "serror_rate": 0.01
        }),
        ("10.0.0.99", {f: 0 for f in FEATURE_NAMES} | {
            "src_bytes": 80000, "dst_bytes": 0,
            "count": 511, "srv_count": 511,
            "serror_rate": 0.99, "same_srv_rate": 1.0
        }),
        ("172.16.5.22", {f: 0 for f in FEATURE_NAMES} | {
            "duration": 0.2, "diff_srv_rate": 0.95,
            "count": 120, "logged_in": 0, "src_bytes": 150
        }),
        ("192.168.1.5", {f: 0 for f in FEATURE_NAMES} | {
            "num_failed_logins": 8, "is_guest_login": 1,
            "logged_in": 0, "src_bytes": 280
        }),
        ("10.10.1.1", {f: 0 for f in FEATURE_NAMES} | {
            "root_shell": 1, "su_attempted": 1,
            "num_shells": 3, "num_root": 5, "hot": 12
        }),
    ]

    for ip, features in test_cases:
        result = ids.predict(features)
        raise_alert(result, src_ip=ip)

    print("\n" + "═"*60)
    print("  PROJECT COMPLETE — IDS DEMO FINISHED")
    print("═"*60)
    print("""
  Techniques covered:
  ┌────────────────────────┬───────────────────────────────────┐
  │ Component              │ Description                       │
  ├────────────────────────┼───────────────────────────────────┤
  │ Dataset                │ 20k synthetic flows (KDD-inspired)│
  │ Random Forest          │ Multi-class attack classification  │
  │ SVM (RBF kernel)       │ Binary normal vs attack           │
  │ Isolation Forest       │ Unsupervised anomaly detection    │
  │ Feature Engineering    │ 30 network-flow features          │
  │ Alert Engine           │ Severity-based alerting           │
  └────────────────────────┴───────────────────────────────────┘
""")


if __name__ == "__main__":
    run_demo()
