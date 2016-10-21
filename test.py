'''
Created on Feb 6, 2014

@author: matrix1
'''
import unittest

import MSP430
from MSP430 import CPU, CodeSequence, N, R, B, R_PC, R_SP, R_SR
from PyCircuit import TestSignal, FullAdder, SRLatch, DLatch, DFlipFlop, \
    intToSignals, signalsToInt, RippleCarryAdder, Vector, Multiplier, \
    Decoder, Memory, ROM, calcAreaDelay, KoggeStoneAdder, \
    TestVector, DecimalAdder, simplify, Case, DontCare, current_cache, Clock, Register, ConstantSignal, ConstantVector, \
    VFalse, monitor


class Test(unittest.TestCase):
    def assertVectorEqual(self, this, other, msg=None):
        if isinstance(this, Vector):
            thisVal = this.toUint()
            bits = len(this)
        else:
            thisVal = this
            bits = 0
        if isinstance(other, Vector):
            otherVal = other.toUint()
            if bits > 0:
                self.assertEqual(len(other), bits, "Vectors not of equal width")
        else:
            otherVal = other
        mask = (1 << bits) - 1
        self.assertEqual(thisVal & mask, otherVal & mask, msg)

    def _test_str(self):
        self.assertEqual(str(TestSignal(0)), 'TestSignal(False)')
        self.assertEqual(str(TestSignal(1)), 'TestSignal(True)')
        self.assertEqual(str(ConstantVector(-10)), 'ConstantVector(-10, 16)')

    def test_SRLatch(self):
        r = TestSignal()
        s = TestSignal()
        signals = SRLatch(s=s, r=r)
        self.assertEqual([sig.value for sig in signals], [False, True])
        s.set()
        self.assertEqual([sig.value for sig in signals], [True, False])
        s.reset()
        self.assertEqual([sig.value for sig in signals], [True, False])
        r.set()
        self.assertEqual([sig.value for sig in signals], [False, True])
        r.reset()
        self.assertEqual([sig.value for sig in signals], [False, True])
        s.set()
        r.set()
        self.assertEqual([sig.value for sig in signals], [False, False])

    def test_DLatch(self):
        d = TestSignal()
        e = TestSignal(True)
        signals = DLatch(d, e)
        self.assertEqual([sig.value for sig in signals], [False, True])
        d.set()
        self.assertEqual([sig.value for sig in signals], [True, False])
        d.reset()
        self.assertEqual([sig.value for sig in signals], [False, True])
        e.reset()
        self.assertEqual([sig.value for sig in signals], [False, True])
        d.set()
        self.assertEqual([sig.value for sig in signals], [False, True])
        e.set()
        self.assertEqual([sig.value for sig in signals], [True, False])
        e.reset()
        self.assertEqual([sig.value for sig in signals], [True, False])

    def test_DFlipFlop(self):
        d = TestSignal()
        clk = TestSignal()
        signals = DFlipFlop(d, clk)
        self.assertEqual([sig.value for sig in signals], [False, True])
        d.set()
        self.assertEqual([sig.value for sig in signals], [False, True])
        clk.set()
        self.assertEqual([sig.value for sig in signals], [True, False])
        clk.reset()
        self.assertEqual([sig.value for sig in signals], [True, False])
        d.reset()
        self.assertEqual([sig.value for sig in signals], [True, False])
        clk.set()
        self.assertEqual([sig.value for sig in signals], [False, True])
        d.set()
        self.assertEqual([sig.value for sig in signals], [False, True])
        clk.reset()
        self.assertEqual([sig.value for sig in signals], [False, True])
        clk.set()
        self.assertEqual([sig.value for sig in signals], [True, False])
        d.reset()
        self.assertEqual([sig.value for sig in signals], [True, False])
        clk.reset()
        self.assertEqual([sig.value for sig in signals], [True, False])
        clk.set()
        self.assertEqual([sig.value for sig in signals], [False, True])

    def test_FullAdder(self):
        a, b, c = TestSignal(), TestSignal(), TestSignal()
        signals = FullAdder(a, b, c)

        self.assertEqual([sig.value for sig in signals], [False, False])
        a.set()  # sum should be 1 (sum, carry)
        self.assertEqual([sig.value for sig in signals], [True, False])
        b.set()  # sum = 2
        self.assertEqual([sig.value for sig in signals], [False, True])
        c.set()  # sum = 3
        self.assertEqual([sig.value for sig in signals], [True, True])

    def test_intConversion(self):
        for i in xrange(-100, 256):
            ls = intToSignals(i, 9)
            j = signalsToInt(ls, True)
            self.assertEqual(i, j)

    def test_RippleCarryAdder(self):
        for a in xrange(-256, 255, 67):
            for b in xrange(-22756, 32767, 1453):
                for c in xrange(2):
                    # current_cache.clear()
                    als = ConstantVector(a, 16)
                    bls = ConstantVector(b, 16)
                    sls, c_out = RippleCarryAdder(als, bls, bool(c))
                    self.assertVectorEqual(sls, a + b + c)
                    self.assertVectorEqual(c_out, a < 0)

    def test_KoggeStoneAdder(self):
        for a in xrange(-256, 255, 67):
            for b in xrange(-22756, 32767, 1453):
                for c in xrange(2):
                    als = ConstantVector(a, 16)
                    bls = ConstantVector(b, 16)
                    sls, c_out = KoggeStoneAdder(als, bls, bool(c))
                    self.assertCurrent(sls)
                    self.assertVectorEqual(sls, a + b + c)
                    self.assertEqual(bool(c_out), a < 0)

    def assertCurrent(self, vec):
        current_cache.clear()
        self.assertEquals(vec.current(), signalsToInt(vec.ls, False))

    def test_DecimalAdder(self):
        for a in xrange(0, 10000, 907):
            d_a = Vector.concat(*[ConstantVector(int(d), 4) for d in reversed("%04u" % a)])
            for b in xrange(0, 10000, 359):
                d_b = Vector.concat(*[ConstantVector(int(d), 4) for d in reversed("%04u" % b)])
                for c in xrange(2):
                    sls, c_out = DecimalAdder(d_a, d_b, bool(c))
                    self.assertCurrent(sls)
                    sum = int("%x" % sls.concat(c_out).toUint())
                    self.assertEqual(sum, a + b + c, "%i != %i (%i + %i + %i)" % (sum, a + b + c, a, b, c))

    def test_SimplifyConst(self):
        for a in xrange(-256, 255, 67):
            for b in xrange(-22756, 32767, 1453):
                als = ConstantVector(a, 16)
                bls = ConstantVector(b, 16)
                sls, c_out = KoggeStoneAdder(als, bls)
                self.assertCurrent(sls)
                sls, c_out = simplify(sls, c_out)
                self.assertTrue(all(isinstance(s, ConstantSignal) for s in sls.ls))
                self.assertCurrent(sls)

                self.assertVectorEqual(sls, a + b)
                self.assertVectorEqual(c_out.ls[0].value, a < 0)

    def test_Simplify(self):
        for a in xrange(-256, 255, 67):
            als = ConstantVector(a, 16)
            bls = TestVector(0, 16)
            cs = TestVector(0, 1)
            sls, c_out = KoggeStoneAdder(als, bls, cs)
            n_ad = calcAreaDelay(als.concat(bls, cs))
            print a, 'Before:', ':', n_ad,

            sls, c_out = simplify(sls, c_out)

            n_ad = calcAreaDelay(als.concat(bls, cs))
            print 'After: :', n_ad

            for b in xrange(-22756, 32767, 1453):
                for c in xrange(2):
                    bls[:] = b
                    cs[:] = c
                    self.assertVectorEqual(sls, a + b + c)

    def test_arith(self):
        self.assertVectorEqual(-TestVector(10), -10)
        self.assertVectorEqual(TestVector(10) + TestVector(-24), 10 + -24)
        self.assertVectorEqual(ConstantVector(10) - ConstantVector(24), 10 - 24)
        self.assertVectorEqual(ConstantVector(0x0f) | ConstantVector(0xf0), 0x0f | 0xf0)
        self.assertVectorEqual(ConstantVector(0x0e) & ConstantVector(0x03), 0x0e & 0x03)
        self.assertVectorEqual(ConstantVector(0x0e) ^ ConstantVector(0x07), 0x0e ^ 0x07)
        self.assertVectorEqual(~ConstantVector(0xff), ~0xff)
        self.assertVectorEqual(abs(ConstantVector(-5)), abs(-5))

    def test_Negate(self):
        for num in xrange(-22756, 32767, 1453):
            als = ConstantVector(num, 16)
            sls = -als
            self.assertVectorEqual(sls, -num)

    def test_Multiplier(self):
        bitlen = 8
        a = TestVector(255, bitlen)
        b = TestVector(255, bitlen)
        m = Multiplier(a, b, False)
        # noinspection PyCallByClass
        print calcAreaDelay(Vector.concat(a, b))
        self.assertEqual(len(m), bitlen * 2)
        for av in xrange(2, 2 ** bitlen, 23):
            a[:] = av
            for bv in xrange(3, 2 ** bitlen, 27):
                b[:] = bv
                self.assertVectorEqual(m, av * bv)

    def test_Decoder(self):
        a = TestVector(0, 8)
        d = Decoder(a)
        for i in xrange(2 ** 8):
            a[:] = i
            self.assertVectorEqual(d, 2 ** i)

    def test_Memory(self):
        clk = Clock()
        SZ = 10
        a = TestVector(0, SZ)
        d = TestVector(0, 16)
        mem_wr = TestVector(0, 1)
        q = Memory(clk, a, d, mem_wr)
        self.assertVectorEqual(q, d)
        self.assertCurrent(q)
        print calcAreaDelay(a.concat(d).concat(mem_wr))
        for i in xrange(2 ** SZ):
            a[:] = i
            self.assertVectorEqual(q, 0)
            self.assertCurrent(q)
        for i in xrange(2 ** SZ):
            d[:] = i
            a[:] = i
            mem_wr[:] = 1
            self.assertVectorEqual(q, d)
            self.assertCurrent(q)
            mem_wr[:] = 0
            self.assertVectorEqual(q, d)
            self.assertCurrent(q)
        for i in xrange(2 ** SZ):
            a[:] = i
            self.assertVectorEqual(q, i)
            self.assertCurrent(q)

    def test_ROM(self):
        clk = Clock()
        size = 10
        a = TestVector(0, size)
        d = TestVector(0, 16)
        mem_wr = VFalse
        q = ROM(clk, a, d, mem_wr, range(2 ** size))
        self.assertCurrent(q)
        for i in xrange(2 ** size):
            a[:] = i
            self.assertVectorEqual(q, i)
            self.assertCurrent(q)

    def _test_AdderDelay(self):
        for bitlen in xrange(2, 65):
            a = TestVector(-1, bitlen)
            b = TestVector(-1, bitlen)
            m, m_c = simplify(*RippleCarryAdder(a.ls, b.ls))
            c = TestVector(-1, bitlen)
            d = TestVector(-1, bitlen)
            n, n_c = simplify(*KoggeStoneAdder(c, d))

            m_ad = calcAreaDelay(a.concat(b))
            print 'RippleCarryAdder:', bitlen, ':', m_ad

            n_ad = calcAreaDelay(c.concat(d))
            print 'KoggeStoneAdder:', bitlen, ':', n_ad
            self.assertLess(m_ad[0], n_ad[0])
            self.assertGreaterEqual(m_ad[1], n_ad[1])

    def test_DontCare(self):
        v = TestVector(0, 4)
        res = Case(v, {(ConstantVector(1, 4), ConstantVector(6, 4), ConstantVector(8, 4)): ConstantVector(13),
                       (ConstantVector(0, 4), ConstantVector(3, 4), ConstantVector(5, 4)): ConstantVector(100)})
        self.assertVectorEqual(res, 100)
        v[:] = 1
        self.assertVectorEqual(res, 13)
        v[:] = 2
        self.assert_(all(type(el.value) is DontCare for el in res.ls))
        self.assertCurrent(res)

    def test_MSP430RegisterFile(self):
        clk = Clock()
        src_reg = TestVector(0, 4)
        dst_reg = TestVector(0, 4)
        src_incr = TestVector(0, 1)
        src_mode = TestVector(0, 2)
        dst_in = TestVector(0, 16)
        sr_in = TestVector(0, 16)
        dst_wr = TestVector(1, 1)
        bw = TestVector(0, 1)
        pc_incr = TestVector(0, 1)
        src_out, dst_out, regs, src_incr_val = MSP430.RegisterFile(pc_incr, src_reg, dst_reg, src_incr, src_mode,
                                                                   dst_in, sr_in,
                                                                   dst_wr, bw,
                                                                   clk)

        for r in xrange(16):
            if r == 3: continue  # register 3 is only constant generator
            dst_reg[:] = r
            dst_in[:] = r * 2 + 4
            self.cycleAndCheck(clk, dst_out, regs, src_incr_val, src_out)
            self.assertVectorEqual(dst_out, r * 2 + 4)

        for r, reg in enumerate(regs):
            if r not in [2, 3]:
                # print 'reg', r, reg
                self.assertCurrent(reg)
                self.assertVectorEqual(reg, r * 2 + 4)

        for r in xrange(16):
            sr_in[:] = r
            self.cycleAndCheck(clk, dst_out, regs, src_incr_val, src_out)
            self.assertVectorEqual(regs[2], r)

        sr_in[:] = 3
        dst_reg[:] = 2
        dst_in[:] = 5
        self.cycleAndCheck(clk, dst_out, regs, src_incr_val, src_out)
        self.assertVectorEqual(regs[2], 5)

        dst_wr[:] = 0
        sr_in[:] = 4
        dst_in[:] = 6
        self.cycleAndCheck(clk, dst_out, regs, src_incr_val, src_out)
        self.assertVectorEqual(regs[2], 5)

        pc_incr[:] = 1
        for r in xrange(16):
            self.cycleAndCheck(clk, dst_out, regs, src_incr_val, src_out)
            self.assertVectorEqual(regs[0], r * 2 + 6)

    def cycleAndCheck(self, clk, dst_out, regs, src_incr_val, src_out):
        clk.cycle()
        self.assertCurrent(src_out)
        self.assertCurrent(dst_out)
        for reg in regs:
            self.assertCurrent(reg)
        self.assertCurrent(src_incr_val)

    def test_CPU_init(self):
        clk = Clock()
        cs = CodeSequence()

        debuglines = CPU(cs.code, clk)

        for reg in debuglines['regs']:
            print reg
            self.assertVectorEqual(reg, 0)

    def test_MOV(self):
        clk = Clock()
        cs = CodeSequence()
        s1, d1 = cs.MOV(N(2 ** 10), R(4))
        s2, d2 = cs.MOV(N(2 ** 10 + 2 ** 6), R(4), B)
        s3, d3 = cs.MOV(R(4), R(5))
        s4, d4 = cs.MOV(N(2 ** 8), +N(2 ** 16 - 4))
        s5, d5 = cs.MOV(+N(2 ** 16 - 4), +N(2 ** 16 - 6))
        s6, d6 = cs.MOV(+N(2 ** 16 - 4), R(6))
        s7, d7 = cs.MOV(+N(2 ** 16 - 6), R(7))

        debuglines = CPU(cs.code, clk)

        CPU_state = {k: debuglines[k] for k in ['state', 'instr']}
        clk.cycle(d1, CPU_state)  # MOV(N(2**10), R(4))
        self.assertVectorEqual(debuglines['regs'][4], ConstantVector(2 ** 10, 16))
        clk.cycle(d2, CPU_state)  # MOV(N(2**6), R(4), B)
        self.assertVectorEqual(debuglines['regs'][4], ConstantVector(2 ** 6, 16))
        clk.cycle(d3, CPU_state)  # MOV(R(4), R(5))
        self.assertVectorEqual(debuglines['regs'][5], ConstantVector(2 ** 6, 16))
        clk.cycle(d4, CPU_state)  # MOV(N(2 ** 8), +N(2 ** 16 - 4))
        clk.cycle(d5, CPU_state)  # MOV(+N(2 ** 16 - 4), +N(2 ** 16 - 6))
        clk.cycle(d6, CPU_state)  # MOV(+N(2 ** 16 - 4), R(6))
        self.assertVectorEqual(debuglines['regs'][6], ConstantVector(2 ** 8, 16))
        clk.cycle(d7, CPU_state)  # MOV(+N(2**6 - 6), R(7))
        self.assertVectorEqual(debuglines['regs'][7], ConstantVector(2 ** 8, 16))

    def two_arg_alu(self, instr, op, test_seq):
        # first test just the ALU
        flags = {f: TestVector(0, 1) for f in MSP430.FLAGS}
        als = TestVector(1)
        bls = TestVector(2)
        res, flags_out = MSP430.alu(MSP430.INSTRS[instr], bls, als, flags, MSP430.BYTE_WORD['WORD'])
        for a, b, fl, flb in test_seq:
            als[:] = a
            bls[:] = b
            self.assertCurrent(res)
            self.assertVectorEqual(res, op(a, b))
            self.assertEqual(set(f for f in flags_out if flags_out[f]), set(fl))
            for f in MSP430.FLAGS:
                flags[f][:] = int(flags_out[f])

        als = TestVector(0, 8)
        bls = TestVector(0, 8)
        res, flags_out = MSP430.alu(MSP430.INSTRS[instr], bls, als, flags, MSP430.BYTE_WORD['BYTE'])
        for a, b, fl, flb in test_seq:
            als[:] = a
            bls[:] = b
            self.assertCurrent(res)
            self.assertVectorEqual(res, op(a, b))
            self.assertEqual(set(f for f in flags_out if flags_out[f]), set(flb))
            for f in MSP430.FLAGS:
                flags[f][:] = int(flags_out[f])

        # then test executing instructions on the alu
        clk = Clock()
        cs = CodeSequence()
        cycles = []
        for a, b, fl, flb in test_seq:
            s0, d0 = cs.MOV(N(a), R(4))
            fn = getattr(cs, instr)
            s1, d1 = fn(N(b), R(4))
            cycles.append(d0 + d1)
        for a, b, fl, flb in test_seq:
            s0, d0 = cs.MOV(N(a), R(5), B)
            fn = getattr(cs, instr)
            s1, d1 = fn(N(b), R(5), B)
            cycles.append(d0 + d1)
        debuglines = CPU(cs.code, clk)
        for a, b, fl, flb in test_seq:
            clk.cycle(cycles.pop(0))
            self.assertVectorEqual(debuglines['state'], MSP430.STATES['FETCH'])
            self.assertVectorEqual(debuglines['regs'][4], op(a, b),
                              'Result wrong while testing %s %i %i: is %s, should be %s' % (instr,
                                                                                            a, b,
                                                                                            int(debuglines['regs'][4]),
                                                                                            op(a, b)))
            self.assertVectorEqual(debuglines['regs'][2], sum(2 ** MSP430.FLAGS_SR_BITS[f] for f in fl),
                              'Flags wrong while testing %s %i %i: is %s, should be %s' % (instr,
                                                                                           a, b,
                                                                                           int(debuglines['regs'][2]),
                                                                                           sum(2 **
                                                                                               MSP430.FLAGS_SR_BITS[f]
                                                                                               for f in fl)))
        for a, b, fl, flb in test_seq:
            clk.cycle(cycles.pop(0))
            self.assertVectorEqual(debuglines['state'], MSP430.STATES['FETCH'])
            self.assertVectorEqual(debuglines['regs'][5][:8], op(a, b),
                              'Result wrong while testing (byte) %s %i %i: is %s, should be %s' % (instr,
                                                                                                   a, b, int(
                                  debuglines['regs'][5][:8]), op(a, b)))
            self.assertVectorEqual(debuglines['regs'][2], sum(2 ** MSP430.FLAGS_SR_BITS[f] for f in flb),
                              'Flags wrong while testing (byte) %s %i %i: is %s, should be %s' % (instr,
                                                                                                  a, b, int(
                                  debuglines['regs'][2]), sum(2 ** MSP430.FLAGS_SR_BITS[f] for f in flb)))

    def test_ADD(self):
        test_seq = [(1, 2, '', ''), (-1, 2, 'c', 'c'), (1, -2, 'n', 'n'), (2, -2, 'cz', 'cz'),
                    (3751, -4297, 'n', 'n'), (-2 ** 15, 2 ** 15, 'vcz', 'z'), (-2 ** 7, 2 ** 7, 'cz', 'vcz'),
                    (2 ** 15 - 2, 2 ** 15 - 1, 'vn', 'cn'), (-2 ** 15 - 2, -2 ** 15 + 1, 'n', 'n')]
        self.two_arg_alu('ADD', lambda a, b: a + b, test_seq)

    def test_ADDC(self):
        test_seq = [(1, 2, '', ''), (-1, 2, 'c', 'c'), (4, 3, '', ''), (2, -2, 'cz', 'cz'),
                    (3857, -4297, 'n', ''), (-2 ** 15 - 2 ** 7, 2 ** 15 + 2 ** 7, 'cz', 'vcz'),
                    (-2 ** 6 - 1, 2 ** 6 + 1, 'c', 'c'), (2 ** 15 - 1, 2 ** 15 - 1, 'vn', 'cn')]
        self.two_arg_alu('ADDC', lambda a, b: a + b + (b % 2), test_seq)

    def test_SUB(self):
        test_seq = [(1, 2, 'c', 'c'), (-1, 3, '', ''), (1, -2, 'nc', 'nc'), (2, -2, 'nc', 'nc'),
                    (3857, -4297, 'nc', 'c'), (-2 ** 15, 2 ** 15, 'vcz', 'z'), (-2 ** 7, 2 ** 7, '', 'vcz'),
                    (2 ** 15 - 2, 2 ** 15 - 1, 'c', 'c'), (-2 ** 15 - 2, -2 ** 15 + 1, 'vc', '')]
        self.two_arg_alu('SUB', lambda a, b: b - a, test_seq)

    def test_SUBC(self):
        test_seq = [(1, 2, 'c', 'c'), (-1, 3, '', ''), (1, -2, 'nc', 'nc'), (3, -3, 'nc', 'nc'),
                    (3857, -4297, 'nc', 'c'), (-2 ** 15 - 2 ** 7 - 2, 2 ** 15 + 2 ** 7 + 1, 'vc', 'vc'),
                    (2 ** 15 - 2, 2 ** 15 - 1, 'c', 'c'), (-2 ** 14 - 2, -2 ** 15 + 1, 'n', '')]
        self.two_arg_alu('SUBC', lambda a, b: b - a + (b % 2), test_seq)

    def test_CMP(self):
        test_seq = [(1, 2, 'c', 'c'), (-1, 3, '', ''), (1, -2, 'nc', 'nc'), (2, -2, 'nc', 'nc'),
                    (3857, -4297, 'nc', 'c'), (-2 ** 15, 2 ** 15, 'vcz', 'z'), (-2 ** 7, 2 ** 7, '', 'vcz'),
                    (2 ** 15 - 2, 2 ** 15 - 1, 'c', 'c'), (-2 ** 15 - 2, -2 ** 15 + 1, 'vc', '')]
        self.two_arg_alu('CMP', lambda a, b: a, test_seq)

    def test_AND(self):
        test_seq = [(1, 2, 'z', 'z'), (-1, 3, 'c', 'c'), (1, -2, 'z', 'z'), (2, -2, 'c', 'c'),
                    (3857, -4297, 'c', 'c'), (-2 ** 15, 2 ** 15, 'nc', 'z'), (-2 ** 7, 2 ** 7, 'c', 'nc'),
                    (2 ** 15 - 2, 2 ** 15 - 1, 'c', 'nc'), (-2 ** 15 - 2, -2 ** 15 + 1, 'z', 'z')]
        self.two_arg_alu('AND', lambda a, b: a & b, test_seq)

    def test_XOR(self):
        test_seq = [(1, 2, 'c', 'c'), (-1, 3, 'nc', 'nc'), (1, -2, 'nc', 'nc'), (2, -2, 'nc', 'nc'),
                    (3857, -4297, 'nc', 'c'), (-2 ** 15, 2 ** 15, 'vz', 'z'), (-2 ** 7, 2 ** 7, 'nc', 'vz'),
                    (2 ** 15 - 2, 2 ** 15 - 1, 'c', 'vc'), (-2 ** 15 - 2, -2 ** 15 + 1, 'nc', 'nc')]
        self.two_arg_alu('XOR', lambda a, b: a ^ b, test_seq)

    def one_arg_alu(self, instr, op, test_seq):
        clk = Clock()
        cs = CodeSequence()
        cycles = []
        for a, fl, flb in test_seq:
            s0, d0 = cs.MOV(N(a), R(4))
            fn = getattr(cs, instr)
            s1, d1 = fn(R(4))
            cycles.append(d0 + d1)
        for a, fl, flb in test_seq:
            s0, d0 = cs.MOV(N(a), R(5), B)
            fn = getattr(cs, instr)
            s1, d1 = fn(R(5), B)
            cycles.append(d0 + d1)
        debuglines = CPU(cs.code, clk)
        for a, fl, flb in test_seq:
            carry = (int(debuglines['regs'][2]) >> MSP430.FLAGS_SR_BITS['c']) % 2
            clk.cycle(cycles.pop(0))
            self.assertVectorEqual(debuglines['state'], MSP430.STATES['FETCH'])
            res = op(a, carry, 16)
            self.assertVectorEqual(debuglines['regs'][4], ConstantVector(res, 16),
                              'Result wrong while testing %s %i: is %s, should be %s' % (instr, a,
                                                                                         int(debuglines['regs'][4]),
                                                                                         res))
            self.assertVectorEqual(debuglines['regs'][2],
                                   ConstantVector(sum(2 ** MSP430.FLAGS_SR_BITS[f] for f in fl), 16),
                              'Flags wrong while testing %s %i: is %s, should be %s' % (instr, a,
                                                                                        int(debuglines['regs'][2]),
                                                                                        sum(2 **
                                                                                            MSP430.FLAGS_SR_BITS[f]
                                                                                            for f in fl)))
        for a, fl, flb in test_seq:
            carry = (int(debuglines['regs'][2]) >> MSP430.FLAGS_SR_BITS['c']) % 2
            clk.cycle(cycles.pop(0))
            self.assertVectorEqual(debuglines['state'], MSP430.STATES['FETCH'])
            res = op(a, carry, 8)
            self.assertVectorEqual(debuglines['regs'][5][:8], ConstantVector(res, 8),
                              'Result wrong while testing (byte) %s %i: is %s, should be %s' % (instr,
                                                                                                a, int(
                                  debuglines['regs'][5][:8]), res))
            self.assertVectorEqual(debuglines['regs'][2],
                                   ConstantVector(sum(2 ** MSP430.FLAGS_SR_BITS[f] for f in flb), 16),
                              'Flags wrong while testing (byte) %s %i: is %s, should be %s' % (instr,
                                                                                               a, int(
                                  debuglines['regs'][2]), sum(2 ** MSP430.FLAGS_SR_BITS[f] for f in flb)))

    def test_RRA(self):
        test_seq = [(1, 'zc', 'zc'), (7, 'c', 'c'), (-2, 'n', 'n'), (1 + 4 + 16 + 64, 'c', 'c'),
                    (2 ** 15 - 1, 'c', 'cn')]
        self.one_arg_alu('RRA', lambda a, c, l: a >> 1, test_seq)

    def test_RRC(self):
        test_seq = [(1, 'zc', 'zc'), (6, 'vn', 'vn'), (5, 'c', 'c'), (1 + 2 + 16 + 64, 'vnc', 'vnc'),
                    (2 ** 15 - 1, 'vnc', 'nc'), (116, 'vn', 'vn'), (0, 'z', 'z')]
        self.one_arg_alu('RRC', lambda a, c, l: (a >> 1) | c << (l - 1), test_seq)

    def test_OneArgInstr(self):
        clk = Clock()
        cs = CodeSequence()
        v = -32768
        s1, d1 = cs.RRA(N(v))
        s2, d2 = cs.MOV(+N(s1), R(4))
        s3, d3 = cs.RRA(+N(s1))
        s4, d4 = cs.MOV(+N(s1), R(5))
        s5, d5 = cs.MOV(N(s1), R(6))
        s6, d6 = cs.RRA(~R(6))
        s7, d7 = cs.MOV(+N(s1), R(6))
        s8, d8 = cs.MOV(R(6), R(7))
        s9, d9 = cs.RRA(R(7))

        debuglines = CPU(cs.code, clk)
        clk.cycle(d1, debuglines)
        clk.cycle(d2, debuglines)
        self.assertVectorEqual(debuglines['regs'][4], ConstantVector(v / 2, 16))
        clk.cycle(d3)
        clk.cycle(d4)
        clk.cycle(d5)
        clk.cycle(d6)
        clk.cycle(d7, debuglines)
        clk.cycle(d8, debuglines)
        clk.cycle(d9, debuglines)
        self.assertVectorEqual(debuglines['regs'][4], ConstantVector(v / 2, 16))
        self.assertCurrent(debuglines['regs'][4])
        self.assertVectorEqual(debuglines['regs'][5], ConstantVector(v / 4, 16))
        self.assertVectorEqual(debuglines['regs'][6], ConstantVector(v / 8, 16))
        self.assertVectorEqual(debuglines['regs'][7], ConstantVector(v / 16, 16))

    def test_PUSH(self):
        clk = Clock()
        cs = CodeSequence()
        s0, d0 = cs.MOV(N(0xFFFF), R(1))
        s1, d1 = cs.PUSH(R(1))
        s2, d2 = cs.MOV(~R(1), R(4))
        s3, d3 = cs.PUSH(N(255))
        s4, d4 = cs.MOV(~R(1), R(4))
        s5, d5 = cs.PUSH(~R(0))
        s6, d6 = cs.MOV(~R(1), R(4))

        debuglines = CPU(cs.code, clk)
        clk.cycle(d0)
        self.assertVectorEqual(debuglines['regs'][1], ConstantVector(-2, 16))
        clk.cycle(d1)
        self.assertVectorEqual(debuglines['regs'][1], ConstantVector(-4, 16))
        clk.cycle(d2)
        self.assertVectorEqual(debuglines['regs'][4], ConstantVector(-2, 16))
        clk.cycle(d3)
        self.assertVectorEqual(debuglines['regs'][1], ConstantVector(-6, 16))
        clk.cycle(d4)
        self.assertVectorEqual(debuglines['regs'][4], ConstantVector(255, 16))
        clk.cycle(d5)
        clk.cycle(d6)
        self.assertVectorEqual(debuglines['regs'][4], ConstantVector(16676, 16))

    def test_RETI(self):
        clk = Clock()
        cs = CodeSequence()
        s0, d0 = cs.MOV(N(0xFFFF), R(R_SP))
        s1, d1 = cs.EINT()
        s2, d2 = cs.PUSH(R(R_PC))
        s3, d3 = cs.PUSH(R(R_SR))
        s4, d4 = cs.DINT()
        s5, d5 = cs.RETI()

        debuglines = CPU(cs.code, clk)
        clk.cycle(d0)
        self.assertVectorEqual(debuglines['regs'][R_SP], ConstantVector(-2, 16))
        clk.cycle(d1)
        self.assertVectorEqual(debuglines['regs'][R_SR], ConstantVector(8, 16))
        clk.cycle(d2)
        clk.cycle(d3)
        clk.cycle(d4)
        self.assertVectorEqual(debuglines['regs'][R_SR], ConstantVector(0, 16))
        clk.cycle(d5, debuglines)
        self.assertVectorEqual(debuglines['regs'][R_SR], ConstantVector(8, 16))
        self.assertVectorEqual(debuglines['regs'][R_PC], ConstantVector((s0 + s1) * 2, 16))

    def test_BRANCH(self):
        clk = Clock()
        cs = CodeSequence()
        s0, d0 = cs.MOV(N(0xFFFF), R(10))
        s1, d1 = cs.BRANCH(R(4))

        debuglines = CPU(cs.code, clk)
        for _ in range(2):
            clk.cycle(d0)
            clk.cycle(d1)
            self.assertVectorEqual(debuglines['regs'][0], 0)

    def test_JMP(self):
        clk = Clock()
        cs = CodeSequence()
        l1 = cs.label()
        s0, d0 = cs.MOV(N(0xFFFF), R(10))
        s1, d1 = cs.JMP(l1)

        debuglines = CPU(cs.code, clk)
        for _ in range(2):
            clk.cycle(d0)
            clk.cycle(d1)
            self.assertVectorEqual(debuglines['regs'][0], 0)

    def test_JZ(self):
        clk = Clock()
        cs = CodeSequence()
        l1 = cs.label()
        s0, d0 = cs.CLRZ()
        s1, d1 = cs.JZ(l1)
        s2, d2 = cs.SETZ()
        s3, d3 = cs.JZ(l1)

        debuglines = CPU(cs.code, clk)
        for _ in range(2):
            clk.cycle(d0)
            clk.cycle(d1)
            clk.cycle(d2)
            clk.cycle(d3)
            self.assertVectorEqual(debuglines['regs'][0], 0)

    def test_JN(self):
        clk = Clock()
        cs = CodeSequence()
        l1 = cs.label()
        s0, d0 = cs.CLRN()
        s1, d1 = cs.JN(l1)
        s2, d2 = cs.SETN()
        s3, d3 = cs.JN(l1)

        debuglines = CPU(cs.code, clk)
        for _ in range(2):
            clk.cycle(d0)
            clk.cycle(d1)
            clk.cycle(d2)
            clk.cycle(d3)
            self.assertVectorEqual(debuglines['regs'][0], 0)

    def test_JC(self):
        clk = Clock()
        cs = CodeSequence()
        l1 = cs.label()
        s0, d0 = cs.CLRC()
        s1, d1 = cs.JC(l1)
        s2, d2 = cs.SETC()
        s3, d3 = cs.JC(l1)

        debuglines = CPU(cs.code, clk)
        for _ in range(2):
            clk.cycle(d0)
            clk.cycle(d1)
            clk.cycle(d2)
            clk.cycle(d3)
            self.assertVectorEqual(debuglines['regs'][0], 0)

    def test_CALL(self):
        clk = Clock()
        cs = CodeSequence()
        s0, d0 = cs.MOV(N(0xFFFF), R(1))
        s1, d1 = cs.MOV(~R(1), R(4))
        s2, d2 = cs.CALL(N(s0 * 2))

        debuglines = CPU(cs.code, clk)
        CPU_state = {k: debuglines[k] for k in ['state', 'instr']}
        clk.cycle(d0, CPU_state)
        self.assertVectorEqual(debuglines['regs'][1], -2)
        clk.cycle(d1, CPU_state)
        self.assertVectorEqual(debuglines['regs'][4], 0)
        clk.cycle(d2, CPU_state)
        self.assertVectorEqual(debuglines['regs'][0], s0 * 2)
        self.assertVectorEqual(debuglines['regs'][1], -4, 16)
        clk.cycle(d1, CPU_state)
        self.assertVectorEqual(debuglines['regs'][4], (s0 + s1 + s2) * 2)

    def test_Register(self):
        clk = Clock()
        r = Register(clk, 1, 2)
        res, c = KoggeStoneAdder(r, ConstantVector(1, 2))
        r.connect(res)
        for _ in range(8):
            s = res.current()
            self.assertCurrent(r)
            self.assertCurrent(res)
            clk.cycle()
            self.assertEqual(r.current(), s)

    def _test_SimplifyCPU(self):
        clk = Clock()
        cs = CodeSequence()
        debuglines = CPU(cs.code, clk)
        print debuglines
        mem_out = debuglines['mem_out']
        before = calcAreaDelay(mem_out.ls)
        from sys import setrecursionlimit
        setrecursionlimit(int(1e3))
        simplify(mem_out)
        print before, calcAreaDelay(mem_out.ls)

    def test_DontCareDecoder(self):
        a = TestVector(0, 4)
        monitor(a)
        d = Decoder(a)
        monitor(d.ls)
        for m in xrange(2 ** 4):
            for i in xrange(2 ** 4):
                a[:] = DontCare(i, m)
                # self.assertVectorEqual(d, 2 ** i)
                self.assertCurrent(d)

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
