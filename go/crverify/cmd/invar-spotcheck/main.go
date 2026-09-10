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
	"runtime/pprof"
	"time"

	"github.com/anomly-labs/invar/go/crverify"
)

func main() {
	gguf := flag.String("gguf", "", "b-posit8 GGUF")
	dump := flag.String("dump", "", "INVAR_LOGITS_OUT dump (JSON lines)")
	rows := flag.Int("rows", 256, "challenged rows per evaluation")
	elementwise := flag.Bool("elementwise", false, "also re-execute every layer's elementwise ops (rmsnorm, rope, swiglu, residual adds)")
	units := flag.Bool("units", false, "also re-execute every layer's matmul units (dump made with INVAR_LOGITS_MATMULS=1)")
	unitRows := flag.Int("unit-rows", 8, "challenged rows per matmul unit")
	jobs := flag.Int("jobs", 0, "worker goroutines for the unit spot-check (0 = all cores)")
	nonceHex := flag.String("nonce", "", "challenge nonce hex (default random)")
	cpuprof := flag.String("cpuprofile", "", "write a CPU profile to this file (go tool pprof)")
	flag.Parse()
	if *cpuprof != "" {
		pf, err := os.Create(*cpuprof)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
		pprof.StartCPUProfile(pf)
		defer pprof.StopCPUProfile()
	}
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
	if os.Getenv("INVAR_SPOTCHECK_PL") == "raw" {
		if err := crverify.EnableFabric(); err != nil {
			fmt.Fprintln(os.Stderr, "fabric:", err)
			os.Exit(2)
		}
		fmt.Println("exact accumulation: bp8_dot_pe fabric (INVAR_SPOTCHECK_PL=raw), " + crverify.FabricMode())
	}
	g, err := crverify.OpenGGUF(*gguf)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	// The per-layer parse is a superset of the lm_head one, so when -units is set read the
	// dump ONCE and derive the lm_head evals from it. Parsing 70 MB of hex twice was the
	// dominant cost on a slow board.
	f, err := os.Open(*dump)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	var evals []crverify.Eval
	var uevals []crverify.UnitEval
	tp := time.Now()
	if *units {
		uevals, err = crverify.ReadDumpUnits(f)
		evals = make([]crverify.Eval, len(uevals))
		for i, u := range uevals {
			evals[i] = crverify.Eval{Hidden: u.Hidden, Logits: u.Logits}
		}
	} else {
		evals, err = crverify.ReadDump(f)
	}
	f.Close()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	parse := time.Since(tp)
	t0 := time.Now()
	res := crverify.VerifyDumpParallel(g, evals, nonce, *rows, *jobs)
	fmt.Printf("nonce %s rows/eval %d evals %d — %s (parse %.2fs, check %.2fs)\n",
		hex.EncodeToString(nonce), *rows, len(evals), res.Why, parse.Seconds(), time.Since(t0).Seconds())
	if res.OK && *units {
		t1 := time.Now()
		ures, _ := crverify.VerifyUnitsParallel(g, uevals, append(append([]byte{}, nonce...), 'u'), *unitRows, *jobs)
		fmt.Printf("units — %s (%.2fs)\n", ures.Why, time.Since(t1).Seconds())
		if u, p, n := crverify.UnitCoverage(g, uevals, *unitRows); u != "" {
			fmt.Printf("coverage — %d of %d rows per evaluation on the widest unit (%s), %d evaluations: "+
				"a single substituted row is caught with probability %.2f. Raise -unit-rows for more.\n",
				*unitRows, n, u, len(uevals), p)
		}
		res.OK = res.OK && ures.OK
		if res.OK && *elementwise {
			t2 := time.Now()
			eres, _ := crverify.VerifyElementwise(g, uevals)
			fmt.Printf("elementwise — %s (%.2fs)\n", eres.Why, time.Since(t2).Seconds())
			res.OK = res.OK && eres.OK
		}
	} else if *elementwise && !*units {
		// genuine misuse; a failed earlier check must fall through to REJECT, not to this
		fmt.Fprintln(os.Stderr, "-elementwise requires -units (both read the per-layer dump)")
		os.Exit(2)
	}
	if d, t := crverify.FabricStats(); d > 0 {
		fmt.Printf("fabric — bp8_dot_pe did %d dot products in %d DMA transfers\n", d, t)
	}
	pprof.StopCPUProfile()
	if res.OK {
		fmt.Println("ACCEPT")
		os.Exit(0)
	}
	fmt.Println("REJECT")
	os.Exit(1)
}
