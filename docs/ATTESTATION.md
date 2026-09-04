Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.

# Attestation-bound, hardware-signed worldlines

A worldline already proves *what was computed* and that the record was not edited.
Two things it did not prove until now: *which platform* it was computed on, and
*which device* wrote it. This page adds both, honestly scoped.

- **Binding** ties the chain to the platform's attestation evidence. Genesis is derived
  from the evidence, and every receipt certifies it. The chain cannot be read under
  any other evidence.
- **Signing** makes every entry carry a signature from a key that lives in a TPM (or,
  stated as such, in a software key file). A modified image cannot mint receipts when
  the key is bound to a PCR policy.

Neither makes the computation confidential. If you need that, run INVAR inside a
confidential VM (Intel TDX, AMD SEV-SNP) with an NVIDIA GPU in confidential mode, and
bind to *that* attestation. See the trust boundary at the end.

## 1. What INVAR does and does not verify

INVAR does **not** re-implement SEV-SNP, TDX, or NVIDIA GPU attestation verification.
The vendors and the Confidential Computing Consortium ship verifiers for that
(`snpguest`, `tdx-guest`, NVIDIA `nvtrust` / NRAS, Veraison, Intel Trust Authority).
You run one of those, keep the evidence bytes and its verdict, and hand both to INVAR.
INVAR records their digests, binds the chain to them, and certifies the binding in
every receipt. A verifier later checks the same evidence against the same digests and
re-runs the vendor verifier if they want to.

## 2. Bind a worldline to platform evidence

```
# evidence you already collected with the vendor tooling, e.g.
#   snpguest report report.bin request.bin       (AMD SEV-SNP guest)
#   tdx-guest quote > quote.bin                  (Intel TDX guest)
#   nvtrust / NRAS EAT token                     (NVIDIA CC GPU)
#   tpm2_quote ... -o quote.bin                  (TPM 2.0, needs /dev/tpmrm0)
invar attest bind --kind sev-snp-report --evidence report.bin \
                  --verifier snpguest --verdict verdict.json --out binding.json
```

`kind` names what the evidence is: `sev-snp-report`, `tdx-quote`, `nvidia-eta`,
`tpm-quote`, `openpcc-bundle`, or `tpm-pcr-bank` (below). The binding file holds
the evidence digest, the verdict digest, a nonce, and the derived genesis:

```
genesis = sha256("invar-genesis-v1" || evidence_digest || nonce)
```

Serve under it:

```
invar serve --model llama3.2 --attest binding.json --signer tpm2:sha256:0,7
```

Every receipt now certifies `computation.host_attestation = {kind, evidence_digest,
nonce, verifier, verdict_digest}` and the first entry's `prev_chain` is the bound
genesis. Verify with the same binding:

```
invar verify worldline.jsonl --attest binding.json --trust-key sha256:<key_id> --require-signature
```

Without `--attest`, entry 0 reports `chain broken`, which is the intended signal that
the chain is bound to evidence you have not supplied. With the wrong platform's
evidence, every entry rejects with `host attestation differs`.

### No TEE? Bind to the TPM PCR bank

Any Linux box with a TPM 2.0 exposes its measured-boot PCR values read-only in sysfs,
no special group needed:

```
invar attest collect-pcrs --out pcr-bank.json         # PCR0-7, SHA-256 bank
invar attest bind --kind tpm-pcr-bank --evidence pcr-bank.json --verifier sysfs-unsigned --out binding.json
```

This is real measured-boot state, but it is **unsigned**: the kind says so. A
`tpm2_quote` (needs read/write on `/dev/tpmrm0`, usually the `tss` group) is the
signed form and should be used when available.

## 3. Sign every entry

| `--signer` | Key | What a signature proves |
|---|---|---|
| `software` | Ed25519 in `~/.invar/signing.key` (0600) | the holder of that file signed it. Honest software anchor. |
| `tpm2` | ECDSA P-256 created inside the TPM, `fixedtpm\|fixedparent\|sensitivedataorigin\|sign`; the private half never leaves the chip | this TPM signed it |
| `tpm2:sha256:0,7` | same key under a PCR policy | this TPM signed it **while PCR0 and PCR7 had the values at key creation**; a different bootloader, firmware, or Secure Boot state cannot sign |

The signature block sits beside the entry, never inside the certified manifest:

```json
"signature": {"backend": "tpm2-ecdsa-p256", "alg": "ECDSA-P256-SHA256",
              "key_id": "sha256:…", "pubkey_pem": "-----BEGIN PUBLIC KEY-----…",
              "signed": "chain", "sig": "<base64 DER>", "pcr_policy": "sha256:0,7"}
```

It signs the entry's `chain` digest, which already commits to the certificate and to
every prior entry. Verifiers pin device keys with `--trust-key` (repeatable) and can
insist on signatures with `--require-signature`. The TPM path uses `tpm2-tools`
(`INVAR_TPM2_BIN` if not on PATH) and needs access to `/dev/tpmrm0`.

## 4. Ledger

The Ledger verifies signatures at the door when entries carry them: a bad or
mismatched signature is a 422 like any other tamper. `LEDGER_TRUSTED_KEYS` (comma-
separated key_ids) pins which device keys the fleet accepts.

## 5. Trust boundary, plainly

- Binding proves the chain was started under specific evidence and that no receipt was
  re-homed. It proves the evidence is genuine only to the extent the vendor verifier
  you ran does. INVAR records that verifier's verdict; it does not stand in for it.
- A TPM-signed receipt proves which TPM signed and, with a PCR policy, in which measured
  state. It does not prove the accelerator computed correctly; re-execution does that.
- A software signature proves possession of a file. It is labelled as such.
- The PCR-bank binding is unsigned evidence. It is labelled as such.
- None of this hides prompts from the operator of the machine. Confidentiality is a
  TEE property; INVAR composes with one, it does not replace one.
