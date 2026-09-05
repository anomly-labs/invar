// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
// Licensed under the Apache License, Version 2.0 (same as the invar repository).

// invar-spotcheck: re-execute challenged lm_head rows of an exact-profile dump in Go.
//
//	invar-spotcheck -gguf model.gguf -dump logits.jsonl [-rows 256] [-nonce hex]
package main

import (
	"crypto/rand"
	"encoding/hex"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/anomly-labs/invar/go/crverify"
)

func main() {
	gguf := flag.String("gguf", "", "b-posit8 GGUF")
	dump := flag.String("dump", "", "INVAR_LOGITS_OUT dump (JSON lines)")
	rows := flag.Int("rows", 256, "challenged rows per evaluation")
	units := flag.Bool("units", false, "also re-execute every layer's matmul units (dump made with INVAR_LOGITS_MATMULS=1)")
	unitRows := flag.Int("unit-rows", 8, "challenged rows per matmul unit")
	nonceHex := flag.String("nonce", "", "challenge nonce hex (default random)")
	flag.Parse()
	if *gguf == "" || *dump == "" {
		fmt.Fprintln(os.Stderr, "usage: invar-spotcheck -gguf model.gguf -dump logits.jsonl [-rows N] [-nonce hex]")
		os.Exit(2)
	}
	var nonce []byte
	if *nonceHex != "" {
		b, err := hex.DecodeString(*nonceHex)
		if err != nil {
			fmt.Fprintln(os.Stderr, "bad nonce")
			os.Exit(2)
		}
		nonce = b
	} else {
		nonce = make([]byte, 16)
		rand.Read(nonce)
	}
	g, err := crverify.OpenGGUF(*gguf)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	f, err := os.Open(*dump)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	evals, err := crverify.ReadDump(f)
	f.Close()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	t0 := time.Now()
	res := crverify.VerifyDump(g, evals, nonce, *rows)
	fmt.Printf("nonce %s rows/eval %d evals %d — %s (%.2fs)\n", hex.EncodeToString(nonce), *rows, len(evals), res.Why, time.Since(t0).Seconds())
	if res.OK && *units {
		f2, err := os.Open(*dump)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
		uevals, err := crverify.ReadDumpUnits(f2)
		f2.Close()
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
		t1 := time.Now()
		ures, _ := crverify.VerifyUnits(g, uevals, append(append([]byte{}, nonce...), 'u'), *unitRows)
		fmt.Printf("units — %s (%.2fs)\n", ures.Why, time.Since(t1).Seconds())
		res.OK = res.OK && ures.OK
	}
	if res.OK {
		fmt.Println("ACCEPT")
		os.Exit(0)
	}
	fmt.Println("REJECT")
	os.Exit(1)
}
