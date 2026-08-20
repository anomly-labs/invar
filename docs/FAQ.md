Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.

# INVAR — FAQ

**Is this another local-LLM runner?** The running is llama.cpp, credited plainly.
What INVAR adds is the part nothing else has: a re-executable, hash-chained
receipt for every inference, and a team plane that turns those into audit
evidence.

**Do I need an account or the internet?** No. The free agent has no account, no
telemetry, no phone-home. Licenses verify offline. The only network calls are
the ones you configure (your own Ledger).

**What exactly does a receipt prove?** That this runtime binary + these model
weights + this prompt + these parameters produced exactly this output, and that
the record hasn't been altered since — checkable by re-hashing, and by
re-running. It does not prove the answer is correct or the model is good.

**Same prompt on my friend's machine gives a different answer — broken?** No:
the default profile pins reproducibility to a deployment (binary+weights+
config+machine). Cross-machine bit-identical inference is what Anomly's
exact-arithmetic profile adds (it's slower; it's the verification-grade option).

**Can't I just delete the worldline file?** On your own box, yes — you own your
records (that's the point). In a team, agents push to a Ledger the moment each
receipt is created, so the fleet's copy survives local deletion; the Ledger
refuses tampered or rewritten histories at the door.

**Why is the verifier free?** Receipts anyone can't check aren't receipts. The
format is the open Computation Receipts spec with published conformance vectors;
verification is a property of the format, not a feature we sell.

**What hardware do I need?** Whatever runs your model today. INVAR adds
milliseconds of hashing per inference; the model remains the cost. We publish
measured numbers per configuration and name every chip we ship.

**GPU support?** The agent wraps your llama.cpp build — if your llama.cpp uses
your GPU, INVAR does too, receipts included.

**What data leaves my machine?** With no Ledger configured: nothing, ever. With
your Ledger configured: receipts (digests + the evidence texts you chose to
log) go to the server you run, with your token, over your TLS.

**License lost?** Email us from the purchase address; we re-issue. Licenses are
files, not activations — no server outage can brick your install.
