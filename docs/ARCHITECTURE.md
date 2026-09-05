Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.

# INVAR architecture: what each piece proves

One page, top to bottom. Every layer answers one question, and every layer is checkable
by someone who does not trust the layer below it. Nothing here hides data; all of it
proves what happened to data.

```
 client ──prompt──▶ invar serve ──▶ llama.cpp / Ollama ──▶ answer
                       │
                       ├─ receipt        what ran (digests) + certificate            [CR v0.1]
                       ├─ worldline      hash chain of receipts, append-only          [INVAR]
                       ├─ binding        chain genesis ← platform attestation         [attest]
                       ├─ signature      TPM / software key over each entry           [hwsign]
                       ├─ dump digest    (exact profile) commit to logits/matmul rows [spot-check]
                       ├─ statement      COSE_Sign1 over the manifest                 [scitt]
                       └─ Ledger         verify-at-door, exports, transparency log    [ledger/tlog]
 verifier ──▶ re-execute · re-hash · check binding · check signature · spot-check ──▶ verdict (signed)
```

## 1. Receipt — what was computed

A canonical JSON manifest holding digests of the runtime, the model weights, the prompt,
the decode parameters and the output, plus the arithmetic profile. Its SHA-256 is the
certificate. Change any byte of any input and the certificate changes. Open spec
(Computation Receipts v0.1), published conformance vectors, Python, C and Go
implementations that agree byte-for-byte.

Profiles say what re-execution means: `llamacpp-pinned-reexec-v0` and
`ollama-pinned-reexec-v0` reproduce on the pinned deployment; `llamacpp-bposit8-quire-v0`
accumulates every matmul in an exact 256-bit quire and runs every other op through a
deterministic elementwise library, so the whole graph is bit-identical on the CPU and on
CUDA (DETERMINISTIC-GRAPH.md) and every matmul re-executes under the Python and Go
verifiers.

## 2. Worldline — that the record was not edited

Entries are chained: each carries the previous entry's chain digest, and its own chain
digest is SHA-256(previous ‖ certificate). Editing, dropping, reordering or splicing an
entry breaks the chain from that point on. The evidence texts (prompt, output) travel
beside the certified manifest, checked against its digests before any use.

## 3. Binding — on which platform

`invar attest bind` derives the chain's genesis from the platform's attestation evidence
(an SEV-SNP report, a TDX quote, an NVIDIA CC token, a TPM quote, or an unsigned TPM PCR
bank) plus a nonce, and certifies `host_attestation` in every manifest. Receipts cannot be
re-homed to another machine; the chain cannot be read under other evidence. INVAR records
the vendor verifier's verdict digest; it does not reimplement vendor attestation.

## 4. Signature — which device wrote it

Every entry can be signed by a key created inside a TPM 2.0 and never exported, optionally
under a PCR policy so a modified boot cannot mint receipts. A software Ed25519 key is
available and labelled as such. The Ledger verifies signatures at the door and pins fleet
keys.

## 5. Spot-check — that the arithmetic happened (exact profile)

The server commits, in the receipt, to the digest of a dump holding the last-row hidden
state and logits of every evaluation, and optionally the input and output rows of every
matmul in every layer. A verifier picks a challenge afterwards, re-quantises the inputs
exactly as ggml does, re-executes challenged rows against the GGUF weights with an
independent implementation, and compares float32 bits. Three implementations (C kernel,
Python, Go) agree; a 1-ulp change in a served value is caught. Elementwise float ops
(norm, RoPE, SiLU, softmax) stay deployment-pinned.

### 5b. Reference re-execution — that the whole answer follows from the weights and the prompt

Under the exact profile two independent implementations (Python, Go; no llama.cpp code)
replay the certified dump from the weights and the token ids: every layer's rows, the
logits, the greedy chain and its detokenised text are compared with the certified ones,
and the certified prompt text is re-tokenised the way llama.cpp does (byte-level BPE and
sentencepiece). `invar verify --spot-check --units --reexec`. The arithmetic is written
down in EXACT-PROFILE-SPEC.md so a third implementation can be built from the document
and tested against the conformance fixture in `go/crverify/testdata`.

## 6. Statements and the transparency log — that it existed, when, asserted by whom

Any entry, export packet or verification verdict can be enveloped as a SCITT-style Signed
Statement (COSE_Sign1; payload = canonical manifest, subject = certificate). The Ledger
keeps an append-only Merkle log of statements (RFC 6962) and returns inclusion receipts;
consistency proofs show the log only ever appended. Statements carry digests only, so a
public log of computations leaks no prompts and no answers.

## 7. Verdicts — a verifier's conclusion as evidence

`invar verify --verdict-out` signs the verifier's own conclusion (worldline digest,
per-entry verdicts, the checks and challenge used) with the verifier's key. Independent
verifiers' verdicts on the same worldline can be compared (`invar scitt agree`): N-version
verification across implementations, machines, or vendors.

## What INVAR does not do

It does not make computation confidential: that is a TEE or MPC or FHE property. It
composes with a TEE by binding to its attestation. It does not judge whether an answer is
good. Under a float profile it does not claim cross-machine bit-identity. Each boundary is
written down in THREATMODEL.md, per layer, with the tests that exercise it.
