// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
// Licensed under the Apache License, Version 2.0 (same as the invar repository).

// invar-reexec replays an exact-profile dump with the Go reference implementation:
//
//	invar-reexec -gguf model-bposit8.gguf -dump logits.jsonl [-max-evals N]
package main

import (
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/anomly-labs/invar/go/crverify"
)

func main() {
	gguf := flag.String("gguf", "", "b-posit8 GGUF")
	dump := flag.String("dump", "", "INVAR dump with token ids (INVAR_LOGITS_MATMULS=1, no warm-up)")
	maxEvals := flag.Int("max-evals", 0, "replay at most N evaluations")
	flag.Parse()
	if *gguf == "" || *dump == "" {
		fmt.Fprintln(os.Stderr, "usage: invar-reexec -gguf model.gguf -dump logits.jsonl [-max-evals N]")
		os.Exit(2)
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
	defer f.Close()
	t0 := time.Now()
	ok, why, _, _, err := crverify.ReexecDump(g, f, *maxEvals, func(s string) { fmt.Fprintf(os.Stderr, "%s (%.1fs)\n", s, time.Since(t0).Seconds()) })
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	if ok {
		fmt.Printf("ACCEPT — %s in %.1fs\n", why, time.Since(t0).Seconds())
		fmt.Printf("final-argmax: %d\n", crverify.FinalArgmax)
		return
	}
	fmt.Printf("REJECT — %s\n", why)
	os.Exit(1)
}
