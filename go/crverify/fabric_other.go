// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
// Licensed under the Apache License, Version 2.0 (same as the invar repository).

//go:build !(linux && arm64)

package crverify

import "errors"

// EnableFabric: the bp8_dot_pe driver talks to a Zynq UltraScale+ PS; nothing to drive here.
func EnableFabric() error {
	return errors.New("INVAR_SPOTCHECK_PL: the fabric backend is only available on linux/arm64 (Ultra96/KV260)")
}

// FabricMode: no fabric here.
func FabricMode() string { return "" }
