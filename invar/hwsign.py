# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
"""
invar.hwsign — hardware-anchored signatures over worldline entries.

A worldline is tamper-EVIDENT on its own (certificate + hash chain). A signature makes
it tamper-evident *by a particular device*: an entry signed by a key that only exists
inside a TPM, under a policy that dies when the platform's measured state changes, can
only have been minted on that platform in that state. That is the hardware half of
"attestation-bound receipts" (docs/research/verifiable-confidential-ai-2026-09-04.md §4.1).

Two backends, both real, both stated in the signature block so a verifier never has
to guess which it got:

  SoftwareSigner  Ed25519 key in a file. Honest software anchor: proves the signer held
                  the key, nothing about hardware. backend="software-ed25519".
  TPM2Signer      ECDSA P-256 key created inside a TPM 2.0 under the owner hierarchy,
                  attributes fixedtpm|fixedparent|sensitivedataorigin|sign (the private
                  half cannot leave the TPM), optionally bound to a PCR policy so the
                  key refuses to sign once the measured boot state changes. Driven
                  through tpm2-tools (tpm2_createprimary / create / load / sign);
                  backend="tpm2-ecdsa-p256". Needs read/write on /dev/tpmrm0.

Signature block (stored on the entry beside `chain`, never inside the certified manifest):
  {"backend", "alg", "key_id" (sha256 of the SPKI DER), "pubkey_pem",
   "signed" ("chain"), "sig" (base64 DER ECDSA or raw Ed25519), "pcr_policy" (or null)}
verify_signature() re-derives everything from the block + the entry's chain digest.
"""
from __future__ import annotations

import base64
import hashlib
import os
import shutil
import subprocess
import tempfile

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

SOFTWARE = "software-ed25519"
TPM2 = "tpm2-ecdsa-p256"


def _key_id(pub_pem: str) -> str:
    pub = serialization.load_pem_public_key(pub_pem.encode())
    der = pub.public_bytes(serialization.Encoding.DER,
                           serialization.PublicFormat.SubjectPublicKeyInfo)
    return "sha256:" + hashlib.sha256(der).hexdigest()


def _chain_bytes(entry: dict) -> bytes:
    return entry["chain"].encode()


# --------------------------------------------------------------------------- software

class SoftwareSigner:
    backend = SOFTWARE

    def __init__(self, key_path: str):
        self.key_path = key_path
        if os.path.exists(key_path):
            with open(key_path, "rb") as f:
                self._key = serialization.load_pem_private_key(f.read(), None)
        else:
            self._key = ed25519.Ed25519PrivateKey.generate()
            pem = self._key.private_bytes(serialization.Encoding.PEM,
                                          serialization.PrivateFormat.PKCS8,
                                          serialization.NoEncryption())
            fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(pem)
        self.pubkey_pem = self._key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo).decode()
        self.key_id = _key_id(self.pubkey_pem)

    def sign_raw(self, msg: bytes) -> bytes:
        """Raw Ed25519 signature over arbitrary bytes (COSE Sig_structure etc.)."""
        return self._key.sign(msg)

    def sign(self, entry: dict) -> dict:
        sig = self._key.sign(_chain_bytes(entry))
        return {"backend": self.backend, "alg": "Ed25519", "key_id": self.key_id,
                "pubkey_pem": self.pubkey_pem, "signed": "chain",
                "sig": base64.b64encode(sig).decode(), "pcr_policy": None}


# --------------------------------------------------------------------------- tpm2

class TPM2Error(RuntimeError):
    pass


class TPM2Signer:
    """Key lives in the TPM; contexts live in `state_dir` (public blobs only:
    key.pub / key.priv are TPM-wrapped and useless outside this TPM)."""
    backend = TPM2

    def __init__(self, state_dir: str, pcrs: str | None = None,
                 tools_bin: str | None = None, tcti: str | None = None):
        self.state_dir = state_dir
        self.pcrs = pcrs                       # e.g. "sha256:0,7" or None
        self.bin = tools_bin or os.environ.get("INVAR_TPM2_BIN") or os.path.dirname(
            shutil.which("tpm2_sign") or "/usr/bin/tpm2_sign")
        self.env = dict(os.environ)
        self.env["TPM2TOOLS_TCTI"] = (tcti or os.environ.get("TPM2TOOLS_TCTI")
                                      or "device:/dev/tpmrm0")
        os.makedirs(state_dir, mode=0o700, exist_ok=True)
        self._p = lambda n: os.path.join(state_dir, n)
        self._provision()
        with open(self._p("key.pem")) as f:
            self.pubkey_pem = f.read()
        self.key_id = _key_id(self.pubkey_pem)

    def _run(self, tool: str, *args: str, stdin: bytes | None = None) -> bytes:
        exe = os.path.join(self.bin, tool)
        r = subprocess.run([exe, *args], input=stdin, capture_output=True,
                           env=self.env, timeout=60)
        if r.returncode != 0:
            raise TPM2Error(f"{tool} failed: {r.stderr.decode(errors='replace')[-400:]}")
        return r.stdout

    def _provision(self) -> None:
        if os.path.exists(self._p("key.ctx")) and os.path.exists(self._p("key.pem")):
            return
        # primary under the owner hierarchy (deterministic for a given TPM + template)
        self._run("tpm2_createprimary", "-C", "o", "-g", "sha256", "-G", "ecc256",
                  "-c", self._p("primary.ctx"))
        extra = []
        if self.pcrs:
            self._run("tpm2_createpolicy", "--policy-pcr", "-l", self.pcrs,
                      "-L", self._p("policy.dat"))
            extra = ["-L", self._p("policy.dat")]
        self._run("tpm2_create", "-C", self._p("primary.ctx"), "-G", "ecc256",
                  "-g", "sha256", "-u", self._p("key.pub"), "-r", self._p("key.priv"),
                  "-a", "fixedtpm|fixedparent|sensitivedataorigin|sign"
                  + ("" if self.pcrs else "|userwithauth"), *extra)
        self._run("tpm2_load", "-C", self._p("primary.ctx"), "-u", self._p("key.pub"),
                  "-r", self._p("key.priv"), "-c", self._p("key.ctx"))
        self._run("tpm2_readpublic", "-c", self._p("key.ctx"), "-f", "pem",
                  "-o", self._p("key.pem"))

    def sign_raw(self, msg: bytes) -> bytes:
        """ECDSA P-256 over sha256(msg) inside the TPM; returns raw r||s (64 bytes),
        the COSE ES256 wire form."""
        return self._sign_digest(hashlib.sha256(msg).digest())

    def _sign_digest(self, digest: bytes) -> bytes:
        with tempfile.TemporaryDirectory() as td:
            dpath, spath = os.path.join(td, "d.bin"), os.path.join(td, "s.bin")
            with open(dpath, "wb") as f:
                f.write(digest)
            auth = []
            if self.pcrs:
                sess = os.path.join(td, "sess.ctx")
                self._run("tpm2_startauthsession", "--policy-session", "-S", sess)
                self._run("tpm2_policypcr", "-S", sess, "-l", self.pcrs)
                auth = ["-p", f"session:{sess}"]
            try:
                self._run("tpm2_sign", "-c", self._p("key.ctx"), "-g", "sha256",
                          "-d", dpath, "-f", "plain", "-o", spath, *auth)
            finally:
                if self.pcrs:
                    subprocess.run([os.path.join(self.bin, "tpm2_flushcontext"), sess],
                                   env=self.env, capture_output=True)
            with open(spath, "rb") as f:
                return f.read()                # plain = r || s, 32 bytes each

    def sign(self, entry: dict) -> dict:
        raw = self._sign_digest(hashlib.sha256(_chain_bytes(entry)).digest())
        r, s = int.from_bytes(raw[:32], "big"), int.from_bytes(raw[32:], "big")
        der = encode_dss_signature(r, s)
        return {"backend": self.backend, "alg": "ECDSA-P256-SHA256",
                "key_id": self.key_id, "pubkey_pem": self.pubkey_pem,
                "signed": "chain", "sig": base64.b64encode(der).decode(),
                "pcr_policy": self.pcrs}


# --------------------------------------------------------------------------- verify

def verify_signature(entry: dict, trusted_key_ids: set[str] | None = None) -> tuple[bool, str]:
    """(ok, why). Verifies the block on `entry["signature"]` against entry["chain"].
    `trusted_key_ids`, when given, pins which device keys are acceptable — a valid
    signature from an unknown key is a REJECT in a fleet setting."""
    blk = entry.get("signature")
    if not blk:
        return False, "unsigned"
    try:
        pub = serialization.load_pem_public_key(blk["pubkey_pem"].encode())
        if _key_id(blk["pubkey_pem"]) != blk["key_id"]:
            return False, "key_id does not match pubkey"
        if trusted_key_ids is not None and blk["key_id"] not in trusted_key_ids:
            return False, "signer key not trusted"
        if blk.get("signed") != "chain":
            return False, f"unsupported signed field {blk.get('signed')!r}"
        sig = base64.b64decode(blk["sig"])
        msg = _chain_bytes(entry)
        if blk["backend"] == SOFTWARE:
            pub.verify(sig, msg)
        elif blk["backend"] == TPM2:
            pub.verify(sig, msg, ec.ECDSA(hashes.SHA256()))
        else:
            return False, f"unknown backend {blk['backend']!r}"
        return True, f"signature ok ({blk['backend']}"
        + (f", pcr policy {blk['pcr_policy']}" if blk.get("pcr_policy") else "") + ")"
    except InvalidSignature:
        return False, "signature invalid"
    except Exception as e:                       # malformed block
        return False, f"signature malformed: {e}"


def make_signer(spec: str | None, state_dir: str):
    """spec: None | "software" | "tpm2" | "tpm2:sha256:0,7" (PCR-policy-bound)."""
    if not spec:
        return None
    if spec == "software":
        return SoftwareSigner(os.path.join(state_dir, "signing.key"))
    if spec == "tpm2" or spec.startswith("tpm2:"):
        pcrs = spec.split(":", 1)[1] if ":" in spec else None
        return TPM2Signer(os.path.join(state_dir, "tpm2"), pcrs=pcrs)
    raise ValueError(f"unknown signer {spec!r}")
