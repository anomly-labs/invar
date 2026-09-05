// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
// Licensed under the Apache License, Version 2.0 (same as the invar repository).

// invar-verify: structural verification of an INVAR worldline in Go — certificate,
// chain, attestation binding, signatures. Exit 0 = ALL ACCEPT, 1 = any REJECT.
//
//	invar-verify worldline.jsonl [-attest binding.json] [-trust-key sha256:...] [-require-signature]
package main

import (
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/anomly-labs/invar/go/crverify"
)

type multi []string

func (m *multi) String() string     { return strings.Join(*m, ",") }
func (m *multi) Set(s string) error { *m = append(*m, s); return nil }

func main() {
	attest := flag.String("attest", "", "attestation binding JSON")
	require := flag.Bool("require-signature", false, "unsigned entries REJECT")
	var keys multi
	flag.Var(&keys, "trust-key", "accept only this signer key_id (repeatable)")
	// accept the worldline path before or after the flags: Go stops parsing flags at
	// the first positional, so a leading path is simply rotated to the end.
	args := os.Args[1:]
	if len(args) > 0 && !strings.HasPrefix(args[0], "-") {
		args = append(append([]string{}, args[1:]...), args[0])
	}
	if err := flag.CommandLine.Parse(args); err != nil || flag.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "usage: invar-verify worldline.jsonl [-attest binding.json] [-trust-key id] [-require-signature]")
		os.Exit(2)
	}
	path := flag.Arg(0)
	opt := crverify.Options{RequireSignature: *require}
	if *attest != "" {
		b, err := crverify.LoadBinding(*attest)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
		opt.Binding = b
	}
	if len(keys) > 0 {
		opt.TrustedKeyIDs = map[string]bool{}
		for _, k := range keys {
			opt.TrustedKeyIDs[k] = true
		}
	}
	f, err := os.Open(path)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	defer f.Close()
	res, err := crverify.VerifyWorldline(f, opt)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	bad := 0
	for _, r := range res {
		v := "ACCEPT"
		if !r.OK {
			v = "REJECT"
			bad++
		}
		fmt.Printf("entry %d: %s — %s\n", r.Index, v, r.Why)
	}
	if bad == 0 {
		fmt.Printf("ALL ACCEPT (%d entries)\n", len(res))
		os.Exit(0)
	}
	fmt.Printf("%d REJECTED (%d entries)\n", bad, len(res))
	os.Exit(1)
}
