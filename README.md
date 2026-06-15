# 🛡️ Cybersecurity Internship Projects

**Author:** Kunal Singh  
**Internship Organization:** Codec Technologies  
**Projects Completed:** Project 3 & Project 6

---

##  Projects Overview

| Project | Title | File |
|---------|-------|------|
| Project 3 | Intrusion Detection System with Machine Learning | `ids_ml_Kunal_Singh.py` |
| Project 6 | Cryptography Algorithms Implementation | `crypto_toolkit_Kunal_Singh.py` |

---

## 🔍 Project 3 — Intrusion Detection System (IDS) with Machine Learning

###  Objective
Design and implement an IDS that uses machine learning algorithms to detect malicious network activity based on traffic patterns.

###  How It Works
1. **Dataset Generation** — 20,000 synthetic network traffic flows are generated (KDD Cup 99 inspired), covering 5 attack categories.
2. **Model Training** — Three ML models are trained:
   - **Random Forest** — Multi-class attack classification
   - **SVM (RBF Kernel)** — Binary classification (Normal vs Attack)
   - **Isolation Forest** — Unsupervised anomaly/zero-day detection
3. **Alert Engine** — Severity-based alerting system (INFO → LOW → MEDIUM → HIGH → CRITICAL)
4. **Synthetic Traffic Prediction Demo** — 5 simulated traffic cases are classified

### Attack Types Detected

| Attack | Description |
|--------|-------------|
| `normal` | Legitimate network traffic |
| `dos` | Denial of Service (SYN flood, Ping of Death) |
| `probe` | Port/network scanning (Nmap, Nessus) |
| `r2l` | Remote-to-Local exploit (brute-force SSH) |
| `u2r` | User-to-Root privilege escalation |

### Technologies Used
- Python 3
- Scikit-learn (Random Forest, SVM, Isolation Forest)
- NumPy, Pandas
- Matplotlib

### How to Run

```bash
# Install dependencies
pip install numpy pandas scikit-learn matplotlib

# Run the project
python ids_ml_Kunal_Singh.py
```

### Expected Output
```
████████████████████████████████████████████████████████████
   INTRUSION DETECTION SYSTEM — ML-BASED  (PROJECT 3)
████████████████████████████████████████████████████████████

[*] Generating synthetic network traffic dataset …
    Samples : 20,000
    Features: 30

[*] Training IDS models …
  [1/3] Training Random Forest (multi-class) …
  [2/3] Training SVM (binary: normal vs attack) …
  [3/3] Training Isolation Forest (anomaly) …

  EVALUATION RESULTS
  ▶ Random Forest — Multi-class Accuracy: ~99%
  ▶ SVM — Binary Accuracy: ~98%
  ▶ Isolation Forest — Anomaly Detection Accuracy: ~85%

  LIVE TRAFFIC SIMULATION
  ┌──────────────────────────────────────────────────────┐
  │  IDS ALERT ENGINE                                    │
  │  Source IP   : 10.0.0.99                             │
  │  Status      : 🚨 ALERT                              │
  │  Attack Type : dos                                   │
  │  Severity    : HIGH                                  │
  └──────────────────────────────────────────────────────┘
```

###  Skills Learned
- Machine learning for cybersecurity
- Network traffic analysis
- Anomaly detection
- Feature engineering (30 network-flow features)

---

##  Project 6 — Cryptography Algorithms Implementation

###  Objective
Implement popular cryptography algorithms like AES, RSA, and SHA to understand encryption, decryption, and secure communication processes.

###  How It Works
The toolkit implements 5 major cryptographic components:

1. **AES Cipher** — Symmetric encryption in CBC and GCM modes
2. **RSA Cipher** — Asymmetric encryption with digital signatures
3. **Hash Utils** — SHA-256 and SHA-512 hashing with file integrity check
4. **HMAC Utils** — Message authentication with constant-time comparison
5. **Password Hasher** — Secure password storage using PBKDF2 with salt

###  Algorithms Implemented

| Algorithm | Mode/Type | Purpose |
|-----------|-----------|---------|
| AES-256-CBC | Symmetric | Classic encryption |
| AES-256-GCM | Authenticated | Modern encryption with integrity |
| RSA-2048 OAEP | Asymmetric | Secure key exchange / encryption |
| RSA-2048 PSS | Digital Signature | Message signing & verification |
| SHA-256 / SHA-512 | Hash Function | Data integrity & checksums |
| HMAC-SHA256 | MAC | Message authentication |
| PBKDF2-HMAC-SHA256 | Key Derivation | Secure password storage |

###  Technologies Used
- Python 3
- PyCryptodome
- hashlib, hmac (Python standard library)

###  How to Run

```bash
# Install dependencies
pip install pycryptodome

# Run the project
python crypto_toolkit_Kunal_Singh.py
```

###  Expected Output
```
████████████████████████████████████████████████████████████
   CRYPTOGRAPHY ALGORITHMS IMPLEMENTATION — PROJECT 6
████████████████████████████████████████████████████████████

══════════════════════════════════════════════════════════
  AES ENCRYPTION
══════════════════════════════════════════════════════════
[+] Plaintext  : Top-secret internship data
[CBC] Decrypted: Top-secret internship data
[GCM] Auth tag : <base64 tag>
✔  AES-CBC and AES-GCM: PASSED

══════════════════════════════════════════════════════════
  RSA ENCRYPTION & DIGITAL SIGNATURES
══════════════════════════════════════════════════════════
[+] Signature valid on original : True
[+] Signature valid on tampered : False
✔  RSA OAEP + PSS Signatures: PASSED

✔  SHA-256 / SHA-512: PASSED
✔  HMAC: PASSED
✔  PBKDF2 Password Hashing: PASSED

══════════════════════════════════════════════════════════
  ALL TESTS PASSED ✔
══════════════════════════════════════════════════════════
```

###  Skills Learned
- Cryptography fundamentals
- Symmetric and asymmetric encryption
- Digital signatures and verification
- Secure password storage techniques
- Message authentication codes

---

##  Requirements

```
numpy
pandas
scikit-learn
matplotlib
pycryptodome
```

---

## Author

**Kunal Singh**  
Cybersecurity Intern — Codec Technologies  
GitHub: [kunal307singh](https://github.com/kunal307singh)
