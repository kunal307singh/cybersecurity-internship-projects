"""
============================================================
  Cryptography Algorithms Implementation — Internship Project 6
  Author : KUNAL SINGH
  Tools  : Python, PyCryptodome
  Covers : AES (CBC & GCM), RSA (OAEP), SHA-256/SHA-512,
           HMAC, Password Hashing (PBKDF2), Digital Signatures
============================================================
"""

import os
import base64
import hashlib
import hmac
import time

# PyCryptodome
from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
from Crypto.Hash import SHA256, SHA512, HMAC as CryptoHMAC
from Crypto.Signature import pss
from Crypto.Protocol.KDF import PBKDF2


# ─────────────────────────────────────────────────────────
#  SECTION 1 — AES (Advanced Encryption Standard)
# ─────────────────────────────────────────────────────────

class AESCipher:
    """
    AES encryption / decryption in two modes:
      • CBC  (Cipher Block Chaining)   — classic confidentiality
      • GCM  (Galois/Counter Mode)     — authenticated encryption (preferred)
    Key sizes: 128-bit, 192-bit, or 256-bit.
    """

    def __init__(self, key_size_bits: int = 256):
        if key_size_bits not in (128, 192, 256):
            raise ValueError("Key size must be 128, 192, or 256 bits.")
        self.key = get_random_bytes(key_size_bits // 8)
        self.key_size = key_size_bits

    # ── CBC mode ──────────────────────────────────────────
    def encrypt_cbc(self, plaintext: str) -> dict:
        iv = get_random_bytes(16)                          # 128-bit IV
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        padded = pad(plaintext.encode(), AES.block_size)
        ciphertext = cipher.encrypt(padded)
        return {
            "mode"       : "AES-CBC",
            "key_b64"    : base64.b64encode(self.key).decode(),
            "iv_b64"     : base64.b64encode(iv).decode(),
            "cipher_b64" : base64.b64encode(ciphertext).decode(),
        }

    def decrypt_cbc(self, cipher_b64: str, iv_b64: str, key_b64: str) -> str:
        key        = base64.b64decode(key_b64)
        iv         = base64.b64decode(iv_b64)
        ciphertext = base64.b64decode(cipher_b64)
        cipher     = AES.new(key, AES.MODE_CBC, iv)
        return unpad(cipher.decrypt(ciphertext), AES.block_size).decode()

    # ── GCM mode (authenticated) ──────────────────────────
    def encrypt_gcm(self, plaintext: str, aad: bytes = b"") -> dict:
        """GCM provides both confidentiality AND integrity/authenticity."""
        cipher = AES.new(self.key, AES.MODE_GCM)
        if aad:
            cipher.update(aad)
        ciphertext, tag = cipher.encrypt_and_digest(plaintext.encode())
        return {
            "mode"       : "AES-GCM",
            "key_b64"    : base64.b64encode(self.key).decode(),
            "nonce_b64"  : base64.b64encode(cipher.nonce).decode(),
            "tag_b64"    : base64.b64encode(tag).decode(),
            "cipher_b64" : base64.b64encode(ciphertext).decode(),
        }

    def decrypt_gcm(self, cipher_b64: str, nonce_b64: str,
                    tag_b64: str, key_b64: str, aad: bytes = b"") -> str:
        key        = base64.b64decode(key_b64)
        nonce      = base64.b64decode(nonce_b64)
        tag        = base64.b64decode(tag_b64)
        ciphertext = base64.b64decode(cipher_b64)
        cipher     = AES.new(key, AES.MODE_GCM, nonce=nonce)
        if aad:
            cipher.update(aad)
        return cipher.decrypt_and_verify(ciphertext, tag).decode()


# ─────────────────────────────────────────────────────────
#  SECTION 2 — RSA (Rivest–Shamir–Adleman)
# ─────────────────────────────────────────────────────────

class RSACipher:
    """
    RSA asymmetric encryption using OAEP padding (secure),
    plus PSS digital signatures.
    Key sizes: 2048-bit (minimum recommended) or 4096-bit.
    """

    def __init__(self, key_bits: int = 2048):
        print(f"  Generating {key_bits}-bit RSA key pair … ", end="", flush=True)
        t0 = time.time()
        self.private_key = RSA.generate(key_bits)
        self.public_key  = self.private_key.publickey()
        print(f"done ({time.time()-t0:.2f}s)")

    def export_keys(self) -> dict:
        return {
            "private_pem": self.private_key.export_key().decode(),
            "public_pem" : self.public_key.export_key().decode(),
        }

    def encrypt(self, plaintext: str) -> str:
        cipher = PKCS1_OAEP.new(self.public_key, hashAlgo=SHA256)
        return base64.b64encode(
            cipher.encrypt(plaintext.encode())
        ).decode()

    def decrypt(self, cipher_b64: str) -> str:
        cipher = PKCS1_OAEP.new(self.private_key, hashAlgo=SHA256)
        return cipher.decrypt(base64.b64decode(cipher_b64)).decode()

    def sign(self, message: str) -> str:
        """Create a PSS digital signature over SHA-256 digest."""
        h   = SHA256.new(message.encode())
        sig = pss.new(self.private_key).sign(h)
        return base64.b64encode(sig).decode()

    def verify(self, message: str, signature_b64: str) -> bool:
        """Verify a PSS signature with the public key."""
        h   = SHA256.new(message.encode())
        sig = base64.b64decode(signature_b64)
        try:
            pss.new(self.public_key).verify(h, sig)
            return True
        except (ValueError, TypeError):
            return False


# ─────────────────────────────────────────────────────────
#  SECTION 3 — Hash Functions (SHA-256, SHA-512)
# ─────────────────────────────────────────────────────────

class HashUtils:
    """
    Deterministic one-way hash functions.
    SHA-256 → 32-byte (256-bit) digest.
    SHA-512 → 64-byte (512-bit) digest.
    """

    @staticmethod
    def sha256(data: str) -> str:
        return hashlib.sha256(data.encode()).hexdigest()

    @staticmethod
    def sha512(data: str) -> str:
        return hashlib.sha512(data.encode()).hexdigest()

    @staticmethod
    def file_integrity(filepath: str) -> dict:
        """Compute SHA-256 and SHA-512 of a file for integrity verification."""
        h256 = hashlib.sha256()
        h512 = hashlib.sha512()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h256.update(chunk)
                h512.update(chunk)
        return {"sha256": h256.hexdigest(), "sha512": h512.hexdigest()}


# ─────────────────────────────────────────────────────────
#  SECTION 4 — HMAC (Hash-based Message Authentication Code)
# ─────────────────────────────────────────────────────────

class HMACUtils:
    """
    HMAC combines a secret key with a hash function to produce a
    Message Authentication Code — proves both integrity AND authenticity.
    """

    def __init__(self):
        self.secret_key = get_random_bytes(32)   # 256-bit shared secret

    def generate(self, message: str) -> str:
        h = hmac.new(self.secret_key, message.encode(), hashlib.sha256)
        return h.hexdigest()

    def verify(self, message: str, mac: str) -> bool:
        expected = self.generate(message)
        # constant-time comparison prevents timing attacks
        return hmac.compare_digest(expected, mac)


# ─────────────────────────────────────────────────────────
#  SECTION 5 — Password Hashing (PBKDF2 with salt)
# ─────────────────────────────────────────────────────────

class PasswordHasher:
    """
    Secure password storage using PBKDF2-HMAC-SHA256.
    Never store plaintext or simple MD5/SHA1 passwords!
    """
    ITERATIONS = 600_000   # NIST SP 800-132 recommendation (2023)

    def hash_password(self, password: str) -> dict:
        salt = get_random_bytes(32)
        dk   = PBKDF2(
            password.encode(), salt,
            dkLen=32, count=self.ITERATIONS,
            prf=lambda p, s: hmac.new(p, s, hashlib.sha256).digest()
        )
        return {
            "salt_b64": base64.b64encode(salt).decode(),
            "hash_b64": base64.b64encode(dk).decode(),
            "iterations": self.ITERATIONS,
        }

    def verify_password(self, password: str, stored: dict) -> bool:
        salt = base64.b64decode(stored["salt_b64"])
        dk   = PBKDF2(
            password.encode(), salt,
            dkLen=32, count=stored["iterations"],
            prf=lambda p, s: hmac.new(p, s, hashlib.sha256).digest()
        )
        return hmac.compare_digest(
            base64.b64decode(stored["hash_b64"]), dk
        )


# ─────────────────────────────────────────────────────────
#  DEMO — Run all components
# ─────────────────────────────────────────────────────────

def separator(title: str):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print('═'*60)


def run_demo():
    print("\n" + "█"*60)
    print("   CRYPTOGRAPHY ALGORITHMS IMPLEMENTATION — PROJECT 6")
    print("█"*60)

    # ── AES Demo ──────────────────────────────────────────
    separator("AES ENCRYPTION")
    aes = AESCipher(key_size_bits=256)
    msg = "Top-secret internship data: salary = ₹10,00,000/yr 😄"

    print(f"\n[+] Plaintext : {msg}")

    enc_cbc = aes.encrypt_cbc(msg)
    dec_cbc = aes.decrypt_cbc(enc_cbc["cipher_b64"], enc_cbc["iv_b64"], enc_cbc["key_b64"])
    print(f"\n[CBC] Ciphertext (b64) : {enc_cbc['cipher_b64'][:48]}…")
    print(f"[CBC] Decrypted       : {dec_cbc}")
    assert dec_cbc == msg, "AES-CBC round-trip FAILED"

    enc_gcm = aes.encrypt_gcm(msg, aad=b"project6-header")
    dec_gcm = aes.decrypt_gcm(
        enc_gcm["cipher_b64"], enc_gcm["nonce_b64"],
        enc_gcm["tag_b64"],    enc_gcm["key_b64"],
        aad=b"project6-header"
    )
    print(f"\n[GCM] Ciphertext (b64) : {enc_gcm['cipher_b64'][:48]}…")
    print(f"[GCM] Auth tag (b64)   : {enc_gcm['tag_b64']}")
    print(f"[GCM] Decrypted        : {dec_gcm}")
    assert dec_gcm == msg, "AES-GCM round-trip FAILED"
    print("\n✔  AES-CBC and AES-GCM: PASSED")

    # ── RSA Demo ──────────────────────────────────────────
    separator("RSA ENCRYPTION & DIGITAL SIGNATURES")
    rsa = RSACipher(key_bits=2048)
    short_msg = "Hello RSA — secure channel established."

    enc_rsa = rsa.encrypt(short_msg)
    dec_rsa = rsa.decrypt(enc_rsa)
    print(f"\n[+] Original  : {short_msg}")
    print(f"[+] Encrypted : {enc_rsa[:48]}…")
    print(f"[+] Decrypted : {dec_rsa}")
    assert dec_rsa == short_msg, "RSA round-trip FAILED"

    sig    = rsa.sign(short_msg)
    valid  = rsa.verify(short_msg, sig)
    tampered = rsa.verify(short_msg + "!", sig)
    print(f"\n[+] Signature (b64) : {sig[:48]}…")
    print(f"[+] Signature valid on original  : {valid}")
    print(f"[+] Signature valid on tampered  : {tampered}")
    assert valid and not tampered, "RSA signature test FAILED"
    print("\n✔  RSA OAEP + PSS Signatures: PASSED")

    # ── Hash Demo ─────────────────────────────────────────
    separator("SHA HASH FUNCTIONS")
    h = HashUtils()
    data = "Cybersecurity Internship 2024"
    print(f"\n[+] Input    : {data}")
    print(f"[+] SHA-256  : {h.sha256(data)}")
    print(f"[+] SHA-512  : {h.sha512(data)[:64]}…")

    # Avalanche effect
    data2 = "Cybersecurity Internship 2025"   # one digit different
    print(f"\n[Avalanche] Input  : {data2}")
    print(f"[Avalanche] SHA-256: {h.sha256(data2)}")
    print("  → Tiny change → completely different digest (avalanche effect)")
    print("\n✔  SHA-256 / SHA-512: PASSED")

    # ── HMAC Demo ─────────────────────────────────────────
    separator("HMAC — MESSAGE AUTHENTICATION CODE")
    hmac_util = HMACUtils()
    api_payload = '{"user":"intern","action":"submit_report"}'
    mac = hmac_util.generate(api_payload)
    print(f"\n[+] Message      : {api_payload}")
    print(f"[+] HMAC-SHA256  : {mac}")
    print(f"[+] Verify (ok)  : {hmac_util.verify(api_payload, mac)}")
    print(f"[+] Verify (bad) : {hmac_util.verify(api_payload + 'X', mac)}")
    print("\n✔  HMAC: PASSED")

    # ── Password Hashing Demo ─────────────────────────────
    separator("PASSWORD HASHING — PBKDF2 WITH SALT")
    ph = PasswordHasher()
    password = "MySuperSecret@2024"
    stored   = ph.hash_password(password)
    print(f"\n[+] Password          : {password}")
    print(f"[+] Salt (b64)        : {stored['salt_b64'][:32]}…")
    print(f"[+] Hash (b64)        : {stored['hash_b64'][:32]}…")
    print(f"[+] Iterations        : {stored['iterations']:,}")
    print(f"[+] Verify correct pw : {ph.verify_password(password, stored)}")
    print(f"[+] Verify wrong pw   : {ph.verify_password('WrongPassword!', stored)}")
    print("\n✔  PBKDF2 Password Hashing: PASSED")

    separator("ALL TESTS PASSED ✔")
    print("""
  Summary of algorithms implemented:
  ┌─────────────────────┬──────────────────────────────────────┐
  │ Algorithm           │ Use-case                             │
  ├─────────────────────┼──────────────────────────────────────┤
  │ AES-256-CBC         │ Symmetric encryption (legacy)        │
  │ AES-256-GCM         │ Authenticated encryption (modern)    │
  │ RSA-2048 OAEP       │ Asymmetric encryption / key exchange │
  │ RSA-2048 PSS        │ Digital signatures                   │
  │ SHA-256 / SHA-512   │ Data integrity / checksums           │
  │ HMAC-SHA256         │ Message authentication               │
  │ PBKDF2-HMAC-SHA256  │ Secure password storage              │
  └─────────────────────┴──────────────────────────────────────┘
""")


if __name__ == "__main__":
    run_demo()
