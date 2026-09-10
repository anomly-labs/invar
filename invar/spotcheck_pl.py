# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
"""Fabric backend for the spot-check's exact dot products: the bp8_dot_pe overlay
(hardware/bp8dot) on a Zynq UltraScale+ board, driven through AXI DMA.

Enabled by INVAR_SPOTCHECK_PL:
  raw            -- talk to the DMA registers through /dev/mem with no PYNQ (Ultra96/KV260
                    with the overlay already programmed via fpga_manager); needs root
  <path>.bit     -- legacy PYNQ path (pynq.Overlay + pynq.allocate)

The PE returns the 256-bit accumulator; the read-out rounding stays in exact_dot's Python so
the result is the same function, bit for bit. Rows longer than one page of words are sent as
chunks and the chunk accumulators are added mod 2^256 -- exact accumulation is order- and
grouping-independent, so this is the same value the PE would produce in one transfer.

Raw-path lessons from the Ultra96 bring-up (hardware/bp8dot/README.md): the PS-PL port widths
are NOT set by the bitstream (a mismatch hangs or misroutes byte lanes), and every page shared
with the non-coherent HP0 port must be cleaned to the point of coherency after the CPU has
touched its cached mapping -- including the fault-in -- or a dirty line evicts over the DMA
data at a random later moment (measured 13/60 wrong results without the flush)."""
import ctypes
import fcntl
import mmap
import os
import struct
import subprocess

DMA_BASE = int(os.environ.get("INVAR_PL_DMA_BASE", "0xA0000000"), 0)
MM2S_DMACR, MM2S_DMASR, MM2S_SA, MM2S_SA_MSB, MM2S_LENGTH = 0x00, 0x04, 0x18, 0x1C, 0x28
S2MM_DMACR, S2MM_DMASR, S2MM_DA, S2MM_DA_MSB, S2MM_LENGTH = 0x30, 0x34, 0x48, 0x4C, 0x58
PAGE = 4096
WORDS_PER_XFER = PAGE // 4
MASK256 = (1 << 256) - 1

stats = {"dots": 0, "transfers": 0, "words": 0}
_backend = None


def available() -> bool:
    return bool(os.environ.get("INVAR_SPOTCHECK_PL"))


_reported = {"dots": 0, "transfers": 0}


def describe() -> str:
    """Suffix for the verdict text when the fabric did the accumulation: counts the dot
    products since the previous call, so each check reports only its own work."""
    d, t = stats["dots"] - _reported["dots"], stats["transfers"] - _reported["transfers"]
    _reported.update(dots=stats["dots"], transfers=stats["transfers"])
    if d == 0:
        return ""
    return f" (exact accumulation in fabric: bp8_dot_pe, {d} dot products, {t} DMA transfers)"


# ---------------------------------------------------------------- raw /dev/mem path
class _Reg:
    def __init__(self, fd, base, size=PAGE):
        self.m = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=base)

    def rd(self, off):
        return struct.unpack_from("<I", self.m, off)[0]

    def wr(self, off, val):
        struct.pack_into("<I", self.m, off, val & 0xFFFFFFFF)


def _phys_of(virt):
    with open("/proc/self/pagemap", "rb") as pm:
        pm.seek((virt // PAGE) * 8)
        e = struct.unpack("<Q", pm.read(8))[0]
    if not (e & (1 << 63)):
        raise RuntimeError("DMA page not present after mlock")
    return (e & ((1 << 55) - 1)) * PAGE


def _flush_lib():
    """libflush.so (dc civac to PoC). Prebuilt next to this file or in ~/.cache/invar; else
    compiled from bp8_flush.c with the system gcc."""
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [os.environ.get("INVAR_PL_FLUSH_SO", ""), os.path.join(here, "libflush.so"),
             os.path.expanduser("~/.cache/invar/libflush.so")]
    src = os.path.join(here, "bp8_flush.c")
    for c in cands:
        if c and os.path.exists(c) and os.path.getmtime(c) >= os.path.getmtime(src):
            return ctypes.CDLL(c)                              # up to date w.r.t. the source
    out = cands[-1]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    subprocess.run(["gcc", "-O2", "-shared", "-fPIC", "-o", out, src], check=True)
    return ctypes.CDLL(out)


class RawPE:
    """One process's handle on the PE: two locked, flushed pages and the DMA register window.
    Transfers are serialised across processes with a lock file so --jobs > 1 stays correct."""

    def __init__(self):
        state = "/sys/class/fpga_manager/fpga0/state"
        if os.path.exists(state) and open(state).read().strip() != "operating":
            raise RuntimeError("PL is not programmed (fpga_manager state != operating)")
        self.libc = ctypes.CDLL("libc.so.6", use_errno=True)
        self.buf = mmap.mmap(-1, PAGE * 2)
        addr = ctypes.addressof(ctypes.c_char.from_buffer(self.buf))
        if self.libc.mlock(ctypes.c_void_p(addr), ctypes.c_size_t(PAGE * 2)) != 0:
            raise RuntimeError("mlock: " + os.strerror(ctypes.get_errno()))
        self.buf[0] = 0; self.buf[PAGE] = 0                  # fault in (dirties one line each)
        self.fl = _flush_lib()
        self.fl.flush_dcache(ctypes.c_void_p(addr), ctypes.c_size_t(PAGE * 2))
        self.p_in, self.p_out = _phys_of(addr), _phys_of(addr + PAGE)
        self.fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)   # uncached alias for all buffer I/O
        self._ports()
        self.dma = _Reg(self.fd, DMA_BASE)
        self.din = _Reg(self.fd, self.p_in)
        self.dout = _Reg(self.fd, self.p_out)
        self.lock = open("/tmp/invar-bp8dot.lock", "w")
        with self._locked():
            self._reset()

    def _ports(self):
        fpd_slcr = _Reg(self.fd, 0xFD615000)                 # AFI_FS [9:8] HPM0: 0=32-bit, as built
        fpd_slcr.wr(0, (fpd_slcr.rd(0) & ~(3 << 8)) | (0 << 8))
        afifm2 = _Reg(self.fd, 0xFD380000)                   # S_AXI_HP0_FPD RD/WR width: 1=64-bit
        for off in (0x00, 0x14):
            afifm2.wr(off, (afifm2.rd(off) & ~3) | 1)

    def _locked(self):
        pe = self

        class L:
            def __enter__(self_):
                fcntl.flock(pe.lock, fcntl.LOCK_EX)

            def __exit__(self_, *a):
                fcntl.flock(pe.lock, fcntl.LOCK_UN)
        return L()

    def _reset(self):
        d = self.dma
        d.wr(MM2S_DMACR, 0x4); d.wr(S2MM_DMACR, 0x4)
        for _ in range(200000):
            if not ((d.rd(MM2S_DMACR) & 0x4) or (d.rd(S2MM_DMACR) & 0x4)):
                return
        raise RuntimeError("AXI DMA reset did not clear -- is the bp8dot overlay loaded and out of reset?")

    def acc_words(self, words) -> int:
        """256-bit accumulator for up to one page of {x,y,se} words."""
        n = len(words)
        assert 0 < n <= WORDS_PER_XFER
        d = self.dma
        with self._locked():
            self.din.m[0:n * 4] = struct.pack(f"<{n}I", *words)
            self.dout.m[0:32] = bytes(32)
            self.fl.barrier()                                  # buffer (Normal-NC) before registers (Device)
            d.wr(S2MM_DMASR, 0x1000); d.wr(MM2S_DMASR, 0x1000)  # W1C the sticky IOC bits first
            d.wr(S2MM_DMACR, 1)
            d.wr(S2MM_DA, self.p_out & 0xFFFFFFFF); d.wr(S2MM_DA_MSB, self.p_out >> 32)
            d.wr(S2MM_LENGTH, 32)
            d.wr(MM2S_DMACR, 1)
            d.wr(MM2S_SA, self.p_in & 0xFFFFFFFF); d.wr(MM2S_SA_MSB, self.p_in >> 32)
            d.wr(MM2S_LENGTH, n * 4)
            for _ in range(2000000):
                if d.rd(S2MM_DMASR) & 0x1000:                  # IOC_Irq = THIS transfer completed
                    break
            else:
                msg = (f"bp8_dot_pe DMA timeout mm2s=0x{d.rd(MM2S_DMASR):08x} "
                       f"s2mm=0x{d.rd(S2MM_DMASR):08x}")
                try:
                    self._reset()       # never exit with a transfer armed on pages we free
                except Exception:
                    pass
                raise RuntimeError(msg)
            self.fl.barrier()                                  # status seen before result read
            limbs = struct.unpack_from("<8I", self.dout.m, 0)
        stats["transfers"] += 1; stats["words"] += n
        return sum(l << (32 * i) for i, l in enumerate(limbs))


# ---------------------------------------------------------------- legacy PYNQ path
class PynqPE:
    def __init__(self, bit):
        from pynq import Overlay, allocate
        import numpy as np
        self.np, self.allocate = np, allocate
        self.dma = Overlay(bit).dma
        self.out = allocate(shape=(8,), dtype=np.uint32)
        self.inb = allocate(shape=(WORDS_PER_XFER,), dtype=np.uint32)

    def acc_words(self, words) -> int:
        n = len(words)
        self.inb[:n] = words; self.inb.flush()
        self.dma.sendchannel.transfer(self.inb[:n]); self.dma.recvchannel.transfer(self.out)
        self.dma.sendchannel.wait(); self.dma.recvchannel.wait(); self.out.invalidate()
        stats["transfers"] += 1; stats["words"] += n
        return sum(int(self.out[i]) << (32 * i) for i in range(8))


def _pe():
    global _backend
    if _backend is None:
        sel = os.environ["INVAR_SPOTCHECK_PL"]
        _backend = RawPE() if sel == "raw" else PynqPE(sel)
    return _backend


def exact_acc_pl(xblocks, yblocks, qfrac: int = 96) -> int:
    """256-bit accumulator of the exact dot, computed in fabric. Same contract as the Python
    accumulator in exact_dot before read-out (mod 2^256, two's complement)."""
    pe = _pe()
    words = []
    for (sx, xq), (sy, yq) in zip(xblocks, yblocks):
        se = (sx + sy + qfrac) & 0xFFFF
        words.extend(((xc << 24) | (yc << 16) | se) for xc, yc in zip(xq, yq))
    acc = 0
    for i in range(0, len(words), WORDS_PER_XFER):
        acc = (acc + pe.acc_words(words[i:i + WORDS_PER_XFER])) & MASK256
    stats["dots"] += 1
    return acc
