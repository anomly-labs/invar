// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
// Licensed under the Apache License, Version 2.0 (same as the invar repository).

package crverify

import "math"

func mathFloat32bits(f float32) uint32     { return math.Float32bits(f) }
func mathFloat32frombits(u uint32) float32 { return math.Float32frombits(u) }
