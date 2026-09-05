// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
// Licensed under the Apache License, Version 2.0 (same as the invar repository).

// invar-statement verifies an INVAR Signed Statement (COSE_Sign1) without Python:
//
//	invar-statement -key signer.pem [-issuer did:web:x] statement.cose
//
// Prints the subject certificate, issuer, key id and the payload kind; exit 0 on ACCEPT.
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/anomly-labs/invar/go/crverify"
)

func main() {
	key := flag.String("key", "", "verifier/signer public key PEM (required)")
	issuer := flag.String("issuer", "", "expected CWT issuer (optional)")
	flag.Parse()
	if *key == "" || flag.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "usage: invar-statement -key signer.pem [-issuer ISS] statement.cose")
		os.Exit(2)
	}
	pem, err := os.ReadFile(*key)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	stmt, err := os.ReadFile(flag.Arg(0))
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	st, err := crverify.VerifyStatement(stmt, string(pem), *issuer)
	if err != nil {
		fmt.Printf("REJECT: %v\n", err)
		os.Exit(1)
	}
	kind, _ := st.Manifest["kind"].(string)
	if kind == "" {
		kind, _ = st.Manifest["profile"].(string)
	}
	fmt.Printf("ACCEPT alg=%d issuer=%s kid=%s subject=%s payload=%s\n", st.Alg, st.Issuer, st.KeyID, st.Subject, kind)
}
