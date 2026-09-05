// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
// Licensed under the Apache License, Version 2.0 (same as the invar repository).

package crverify

// detmath: the ggml-det deterministic elementwise library in Go, bit-exact with the C header
// and the Python port (invar/detmath.py). Go float64 arithmetic is IEEE round-to-nearest; the
// compiler may fuse x*y+z on some architectures (arm64), so every product is passed through an
// explicit float64() conversion, which the Go specification defines as a rounding barrier.

import "math"

func dmul(a, b float64) float64 { return float64(a * b) }
func dadd(a, b float64) float64 { return float64(a + b) }
func dsub(a, b float64) float64 { return float64(a - b) }
func ddiv(a, b float64) float64 { return float64(a / b) }

// float32 ops: the double result rounded to float32 is the correctly rounded float32 result
// (double rounding is innocuous for +, -, *, / and sqrt at these precisions).
func fmul(a, b float32) float32 { return float32(float64(a) * float64(b)) }
func fadd(a, b float32) float32 { return float32(float64(a) + float64(b)) }
func fsub(a, b float32) float32 { return float32(float64(a) - float64(b)) }
func fdiv(a, b float32) float32 { return float32(float64(a) / float64(b)) }
func fsqrt(a float32) float32   { return float32(math.Sqrt(float64(a))) }

func detFloor(x float64) float64 {
	t := int64(x)
	r := float64(t)
	if r > x {
		r = float64(t - 1)
	}
	return r
}

// ---------------------------------------------------------------- exact sums (640-bit, radix 352)

const detBigLimbs = 20
const detBigRadix = 352

type detBig [detBigLimbs]uint32

func (a *detBig) addPiece(w int, parts [3]uint32) {
	var c uint64
	for i := 0; i < 3; i++ {
		if w+i >= detBigLimbs {
			return
		}
		t := uint64(a[w+i]) + uint64(parts[i]) + c
		a[w+i] = uint32(t)
		c = t >> 32
	}
	for i := w + 3; c != 0 && i < detBigLimbs; i++ {
		t := uint64(a[i]) + c
		a[i] = uint32(t)
		c = t >> 32
	}
}

func (a *detBig) addScaled(mi uint64, pos int) {
	w, b := pos>>5, uint(pos&31)
	lo, hi := mi, uint64(0)
	if b != 0 {
		lo = mi << b
		hi = mi >> (64 - b)
	}
	a.addPiece(w, [3]uint32{uint32(lo), uint32(lo >> 32), uint32(hi)})
}

// AddSq: acc += v*v exactly; false for a non-finite v
func (a *detBig) AddSq(v float32) bool {
	vd := float64(v)
	if vd == 0 {
		return true
	}
	fb := math.Float32bits(v)
	if (fb>>23)&0xFF == 0xFF {
		return false
	}
	p := dmul(vd, vd)
	bits := math.Float64bits(p)
	mi := (bits & 0xFFFFFFFFFFFFF) | (1 << 52)
	e2 := int((bits>>52)&0x7FF) - 1023
	a.addScaled(mi, e2-52+detBigRadix)
	return true
}

// AddNonNeg: acc += v for a finite non-negative float32
func (a *detBig) AddNonNeg(v float32) {
	fb := math.Float32bits(v)
	e, f := int((fb>>23)&0xFF), uint64(fb&0x7FFFFF)
	if fb&0x7FFFFFFF == 0 || e == 0xFF {
		return
	}
	var mi uint64
	var e2 int
	if e == 0 {
		mi, e2 = f, -126-23
	} else {
		mi, e2 = f|0x800000, e-127-23
	}
	a.addScaled(mi, e2+detBigRadix)
}

func (a *detBig) top() int {
	for i := detBigLimbs - 1; i >= 0; i-- {
		if a[i] != 0 {
			t := 31
			for (a[i]>>uint(t))&1 == 0 {
				t--
			}
			return 32*i + t
		}
	}
	return -1
}

func (a *detBig) bit(j int) uint32 {
	if j < 0 || j >= 32*detBigLimbs {
		return 0
	}
	return (a[j>>5] >> uint(j&31)) & 1
}

// ToDouble: correctly rounded (nearest-even) value S * 2^-352
func (a *detBig) ToDouble() float64 {
	t := a.top()
	if t < 0 {
		return 0
	}
	var mant uint64
	for j := t; j > t-53; j-- {
		mant = mant<<1 | uint64(a.bit(j))
	}
	guard := a.bit(t - 53)
	sticky := uint32(0)
	for j := t - 54; j >= 0 && sticky == 0; j-- {
		sticky |= a.bit(j)
	}
	e := t - 52
	if guard == 1 && (sticky == 1 || mant&1 == 1) {
		mant++
		if mant == 1<<53 {
			mant >>= 1
			e++
		}
	}
	return math.Ldexp(float64(mant), e-detBigRadix)
}

// SumsqF32 is det_sumsq_f32; RmsScale is det_rms_scale.
func SumsqF32(x []float32) float64 {
	var acc detBig
	for _, v := range x {
		if !acc.AddSq(v) {
			return math.Inf(1)
		}
	}
	return acc.ToDouble()
}

func RmsScale(sumsq float64, n int, eps float32) float32 {
	mean := float32(ddiv(sumsq, float64(n)))
	return fdiv(1, fsqrt(fadd(mean, eps)))
}

func RmsNormRow(x, w []float32, eps float32) []float32 {
	scale := RmsScale(SumsqF32(x), len(x), eps)
	out := make([]float32, len(x))
	for i := range x {
		out[i] = fmul(fmul(x[i], scale), w[i])
	}
	return out
}

// ---------------------------------------------------------------- exp / log2 / exp2 / trig

const (
	detInvLn2 = 1.44269504088896338700e+00
	detLn2Hi  = 6.93147180369123816490e-01
	detLn2Lo  = 1.90821492927058770002e-10
	detLn2    = 6.93147180559945286227e-01
)

var detExpC = [14]float64{1.0, 1.0, 5.00000000000000000000e-01, 1.66666666666666657415e-01, 4.16666666666666643537e-02,
	8.33333333333333321769e-03, 1.38888888888888894189e-03, 1.98412698412698412526e-04,
	2.48015873015873015658e-05, 2.75573192239858925110e-06, 2.75573192239858883130e-07,
	2.50521083854417202234e-08, 2.08767569878681002718e-09, 1.60590438368216133364e-10}

func detExpSmall(r float64) float64 {
	p := detExpC[13]
	for i := 12; i >= 0; i-- {
		p = dadd(dmul(p, r), detExpC[i])
	}
	return p
}

func DetExpD(x float64) float64 {
	k := detFloor(dadd(dmul(x, detInvLn2), 0.5))
	r := dsub(dsub(x, dmul(k, detLn2Hi)), dmul(k, detLn2Lo))
	return math.Ldexp(detExpSmall(r), int(k))
}

func DetExpf(x float32) float32 {
	b := math.Float32bits(x)
	if (b>>23)&0xFF == 0xFF {
		if b&0x7FFFFF != 0 {
			return x
		}
		if b>>31 == 1 {
			return 0
		}
		return x
	}
	if x > 88.75 {
		return float32(math.Inf(1))
	}
	if x < -104.0 {
		return 0
	}
	return float32(DetExpD(float64(x)))
}

func DetExp2D(y float64) float64 {
	k := detFloor(dadd(y, 0.5))
	r := dsub(y, k)
	return math.Ldexp(detExpSmall(dmul(r, detLn2)), int(k))
}

func DetLog2D(x float64) float64 {
	bits := math.Float64bits(x)
	e := int((bits>>52)&0x7FF) - 1023
	m := math.Float64frombits((bits & 0xFFFFFFFFFFFFF) | (1023 << 52))
	if m > 1.41421356237309514547e+00 {
		m = dmul(m, 0.5)
		e++
	}
	s := ddiv(dsub(m, 1.0), dadd(m, 1.0))
	s2 := dmul(s, s)
	p := 0.0
	for k := 29; k >= 1; k -= 2 {
		p = dadd(dmul(p, s2), ddiv(1.0, float64(k)))
	}
	lnm := dmul(2.0, dmul(s, p))
	return dadd(float64(e), dmul(lnm, detInvLn2))
}

const (
	detTwoOverPi = 6.36619772367581382433e-01
	detPio2_1    = 1.57079632673412561417e+00
	detPio2_2    = 6.07710050650619224932e-11
	detPio2_3    = 2.02226624879595063154e-21
)

var detSC = [8]float64{1.0, -1.66666666666666657415e-01, 8.33333333333333321769e-03, -1.98412698412698412526e-04,
	2.75573192239858925110e-06, -2.50521083854417202234e-08, 1.60590438368216133364e-10, -7.64716373181981647590e-13}
var detCC = [9]float64{1.0, -5.00000000000000000000e-01, 4.16666666666666643537e-02, -1.38888888888888894189e-03,
	2.48015873015873015658e-05, -2.75573192239858883130e-07, 2.08767569878681002718e-09, -1.14707455977297245139e-11,
	4.77947733238738529744e-14}

func DetSincosD(x float64) (s, c float64) {
	n := detFloor(dadd(dmul(x, detTwoOverPi), 0.5))
	r := dsub(dsub(dsub(x, dmul(n, detPio2_1)), dmul(n, detPio2_2)), dmul(n, detPio2_3))
	r2 := dmul(r, r)
	ps := detSC[7]
	for i := 6; i >= 0; i-- {
		ps = dadd(dmul(ps, r2), detSC[i])
	}
	ps = dmul(ps, r)
	pc := detCC[8]
	for i := 7; i >= 0; i-- {
		pc = dadd(dmul(pc, r2), detCC[i])
	}
	switch int64(n) & 3 {
	case 0:
		return ps, pc
	case 1:
		return pc, -ps
	case 2:
		return -ps, -pc
	default:
		return -pc, ps
	}
}

func RopeFreq(i, nDims int, freqBase float32) float64 {
	L := dmul(DetLog2D(float64(freqBase)), ddiv(dmul(-2.0, float64(i)), float64(nDims)))
	return DetExp2D(L)
}

func RopeSincos(pos float32, i, nDims int, freqBase, freqScale float32) (s, c float32) {
	theta := dmul(dmul(float64(pos), RopeFreq(i, nDims, freqBase)), float64(freqScale))
	sd, cd := DetSincosD(theta)
	return float32(sd), float32(cd)
}

// RopeSincosFF: RoPE with a per-pair frequency factor (Llama 3 rope_freqs).
func RopeSincosFF(pos float32, i, nDims int, freqBase, freqScale, ff float32) (s, c float32) {
	theta := dmul(ddiv(dmul(float64(pos), RopeFreq(i, nDims, freqBase)), float64(ff)), float64(freqScale))
	sd, cd := DetSincosD(theta)
	return float32(sd), float32(cd)
}

func Sigmoidf(x float32) float32 { return fdiv(1, fadd(1, DetExpf(-x))) }
func Siluf(x float32) float32    { return fmul(x, Sigmoidf(x)) }

// RopeRow rotates one head-dim slice in place semantics (returns a new slice).
func RopeRow(x []float32, pos int, nDims int, freqBase, freqScale, attnFactor float32, neox bool, ff []float32) []float32 {
	y := make([]float32, len(x))
	copy(y, x)
	half := nDims / 2
	for i := 0; i < half; i++ {
		var s, c float32
		if ff != nil {
			s, c = RopeSincosFF(float32(pos), i, nDims, freqBase, freqScale, ff[i])
		} else {
			s, c = RopeSincos(float32(pos), i, nDims, freqBase, freqScale)
		}
		c = fmul(c, attnFactor)
		s = fmul(s, attnFactor)
		var i0, i1 int
		if neox {
			i0, i1 = i, i+half
		} else {
			i0, i1 = 2*i, 2*i+1
		}
		x0, x1 := x[i0], x[i1]
		y[i0] = fsub(fmul(x0, c), fmul(x1, s))
		y[i1] = fadd(fmul(x0, s), fmul(x1, c))
	}
	return y
}

func SwigluRow(gate, up []float32) []float32 {
	out := make([]float32, len(gate))
	for i := range gate {
		out[i] = fmul(Siluf(gate[i]), up[i])
	}
	return out
}

func AddRow(a, b []float32) []float32 {
	out := make([]float32, len(a))
	for i := range a {
		out[i] = fadd(a[i], b[i])
	}
	return out
}

// SoftMaxRow: v = x*scale + mask; e = expf(v - max); p = e * (1 / (float32) exact_sum(e))
func SoftMaxRow(x []float32, scale float32, mask []float32) []float32 {
	v := make([]float32, len(x))
	mx := float32(math.Inf(-1))
	for i := range x {
		v[i] = fmul(x[i], scale)
		if mask != nil {
			v[i] = fadd(v[i], mask[i])
		}
		if v[i] > mx {
			mx = v[i]
		}
	}
	var acc detBig
	e := make([]float32, len(x))
	for i := range v {
		e[i] = DetExpf(fsub(v[i], mx))
		acc.AddNonNeg(e[i])
	}
	inv := fdiv(1, float32(acc.ToDouble()))
	for i := range e {
		e[i] = fmul(e[i], inv)
	}
	return e
}
