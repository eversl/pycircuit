'''
Created on Feb 6, 2014

@author: matrix1
'''
import unittest

import MSP430
from MSP430 import CPU, CodeSequence, N, R, B
from PyCircuit import TestSignal, FullAdder, SRLatch, DLatch, DFlipFlop, \
    intToSignals, signalsToInt, RippleCarryAdder, Vector, Multiplier, \
    Decoder, Memory, RegisterFile, calcAreaDelay, KoggeStoneAdder, \
    TestVector, Signal, DecimalAdder, simplify


class Test(unittest.TestCase):
    def _test_str(self):
        self.assertEqual(str(TestSignal(0)), 'TestSignal(False)')
        self.assertEqual(str(TestSignal(1)), 'TestSignal(True)')
        self.assertEqual(str(Vector(-10)), 'Vector(-10, 16)')

    def test_SRLatch(self):
        r = TestSignal()
        s = TestSignal()
        signals = SRLatch(s=s, r=r)
        self.assertEquals([sig.value for sig in signals], [False, True])
        s.set()
        self.assertEquals([sig.value for sig in signals], [True, False])
        s.reset()
        self.assertEquals([sig.value for sig in signals], [True, False])
        r.set()
        self.assertEquals([sig.value for sig in signals], [False, True])
        r.reset()
        self.assertEquals([sig.value for sig in signals], [False, True])
        s.set()
        r.set()
        self.assertEquals([sig.value for sig in signals], [False, False])


    def test_DLatch(self):
        d = TestSignal()
        e = TestSignal(True)
        signals = DLatch(d, e)
        self.assertEquals([sig.value for sig in signals], [False, True])
        d.set()
        self.assertEquals([sig.value for sig in signals], [True, False])
        d.reset()
        self.assertEquals([sig.value for sig in signals], [False, True])
        e.reset()
        self.assertEquals([sig.value for sig in signals], [False, True])
        d.set()
        self.assertEquals([sig.value for sig in signals], [False, True])
        e.set()
        self.assertEquals([sig.value for sig in signals], [True, False])
        e.reset()
        self.assertEquals([sig.value for sig in signals], [True, False])


    def test_DFlipFlop(self):
        d = TestSignal()
        clk = TestSignal()
        signals = DFlipFlop(d, clk)
        self.assertEquals([sig.value for sig in signals], [False, True])
        d.set()
        self.assertEquals([sig.value for sig in signals], [False, True])
        clk.set()
        self.assertEquals([sig.value for sig in signals], [True, False])
        clk.reset()
        self.assertEquals([sig.value for sig in signals], [True, False])
        d.reset()
        self.assertEquals([sig.value for sig in signals], [True, False])
        clk.set()
        self.assertEquals([sig.value for sig in signals], [False, True])
        d.set()
        self.assertEquals([sig.value for sig in signals], [False, True])
        clk.reset()
        self.assertEquals([sig.value for sig in signals], [False, True])
        clk.set()
        self.assertEquals([sig.value for sig in signals], [True, False])
        d.reset()
        self.assertEquals([sig.value for sig in signals], [True, False])
        clk.reset()
        self.assertEquals([sig.value for sig in signals], [True, False])
        clk.set()
        self.assertEquals([sig.value for sig in signals], [False, True])

    def test_FullAdder(self):
        a, b, c = TestSignal(), TestSignal(), TestSignal()
        signals = FullAdder(a, b, c)

        self.assertEquals([sig.value for sig in signals], [False, False])
        a.set()  # sum should be 1 (sum, carry) 
        self.assertEquals([sig.value for sig in signals], [True, False])
        b.set()  # sum = 2
        self.assertEquals([sig.value for sig in signals], [False, True])
        c.set()  # sum = 3
        self.assertEquals([sig.value for sig in signals], [True, True])


    def test_intConversion(self):
        for i in xrange(-100, 256):
            ls = intToSignals(i, 9)
            j = signalsToInt(ls, True)
            self.assertEqual(i, j)


    def test_RippleCarryAdder(self):
        for a in xrange(-256, 255, 67):
            for b in xrange(-22756, 32767, 1453):
                for c in xrange(2):
                    als = intToSignals(a, 16)
                    bls = intToSignals(b, 16)
                    sls, c_out = RippleCarryAdder(als, bls, Signal(c))
                    sum = signalsToInt(sls, True)
                    self.assertEqual(sum, a + b + c)
                    self.assertEqual(c_out.value, a < 0)


    def test_KoggeStoneAdder(self):
        for a in xrange(-256, 255, 67):
            for b in xrange(-22756, 32767, 1453):
                for c in xrange(2):
                    als = intToSignals(a, 16)
                    bls = intToSignals(b, 16)
                    sls, c_out = KoggeStoneAdder(als, bls, Signal(c))
                    sum = signalsToInt(sls, True)
                    self.assertEqual(sum, a + b + c)
                    self.assertEqual(c_out.value, a < 0)

    def test_Current(self):
        for a in xrange(-256, 255, 67):
            for b in xrange(-22756, 32767, 1453):
                for c in xrange(2):
                    als = intToSignals(a, 16)
                    bls = intToSignals(b, 16)
                    sls, c_out = KoggeStoneAdder(als, bls, Signal(c))

                    print sls.current()

                    sum = signalsToInt(sls, True)
                    self.assertEqual(sum, a + b + c)
                    self.assertEqual(c_out.value, a < 0)


    def test_DecimalAdder(self):
        for a in xrange(0, 10000, 907):
            d_a = Vector(dd for d in reversed("%04u" % a) for dd in Vector(int(d), 4))
            for b in xrange(0, 10000, 359):
                d_b = Vector(dd for d in reversed("%04u" % b) for dd in Vector(int(d), 4))
                for c in xrange(2):
                    sls, c_out = DecimalAdder(d_a, d_b, Signal(c))
                    d_sum = sls.concat(c_out)
                    sum = int(
                        ''.join(reversed([str(signalsToInt(d_sum[i:i + 4], False)) for i in xrange(0, len(d_sum), 4)])))
                    self.assertEqual(sum, a + b + c, "%i != %i (%i + %i + %i)" % (sum, a + b + c, a, b, c))

    def test_SimplifyConst(self):
        for a in xrange(-256, 255, 67):
            for b in xrange(-22756, 32767, 1453):
                als = intToSignals(a, 16)
                bls = intToSignals(b, 16)
            sls, c_out = KoggeStoneAdder(als, bls)
            n_ad = calcAreaDelay(als[:] + bls[:])
            print 'Before:', n_ad,

            sls, c_out = simplify(sls, c_out)

            n_ad = calcAreaDelay(als[:] + bls[:])
            print 'After:', n_ad

            sum = signalsToInt(sls, True)
            self.assertEqual(sum, a + b)
            self.assertEqual(c_out.value, a < 0)

    def test_Simplify(self):
        for a in xrange(-256, 255, 67):
            als = intToSignals(a, 16)
            bls = TestVector(0, 16)
            cs = TestSignal(0)
            sls, c_out = KoggeStoneAdder(als, bls, cs)
            n_ad = calcAreaDelay(als[:] + bls[:] + [cs])
            print 'Before:', a, ':', n_ad,

            sls, c_out = simplify(sls, c_out)

            n_ad = calcAreaDelay(als[:] + bls[:] + [cs])
            print 'After: :', n_ad

            for b in xrange(-22756, 32767, 1453):
                for c in xrange(2):
                    bls[:] = b
                    cs.set(c)
                    sum = signalsToInt(sls, True)
                    self.assertEqual(sum, a + b + c)
                    self.assertEqual(c_out.value, a < 0)


    def test_arith(self):
        self.assertEquals(int(-TestVector(10)), -10)
        self.assertEquals(int(TestVector(10) + TestVector(-24)), 10 + -24)
        self.assertEquals(int(Vector(10) - Vector(24)), 10 - 24)
        self.assertEquals(int(Vector(0x0f) | Vector(0xf0)), 0x0f | 0xf0)
        self.assertEquals(int(Vector(0x0e) & Vector(0x03)), 0x0e & 0x03)
        self.assertEquals(int(Vector(0x0e) ^ Vector(0x07)), 0x0e ^ 0x07)
        self.assertEquals(int(~Vector(0xff)), ~0xff)
        self.assertEquals(int(abs(Vector(-5))), abs(-5))

    def test_Negate(self):
        for num in xrange(-22756, 32767, 1453):
            als = Vector(num, 16)
            sls = -als
            neg = signalsToInt(sls, True)
            self.assertEqual(neg, -num)


    def test_Multiplier(self):
        bitlen = 8
        a = TestVector(0, bitlen)
        b = TestVector(0, bitlen)
        m = Vector(Multiplier(a, b, False))
        print calcAreaDelay(a[:] + b[:])
        self.assertEqual(len(m), bitlen * 2)
        for av in xrange(2, 2 ** bitlen, 23):
            a[:] = av
            for bv in xrange(3, 2 ** bitlen, 27):
                b[:] = bv
                self.assertEqual(signalsToInt(m, False), av * bv)


    def test_Decoder(self):
        a = TestVector(0, 8)
        d = Decoder(a)
        for i in xrange(2 ** 8):
            a[:] = i
            self.assertEqual(signalsToInt(Vector(d), False), 2 ** i)


    def test_Memory(self):
        a = TestVector(0, 9)
        d = TestVector(0, 16)
        mem_wr = TestSignal(0)
        q = Memory(a, d, mem_wr)
        print calcAreaDelay(a)
        print calcAreaDelay(d)
        print calcAreaDelay(mem_wr)
        for i in xrange(2 ** 8):
            d[:] = i
            a[:] = i
            mem_wr.set()
            mem_wr.reset()
        for i in xrange(2 ** 8):
            a[:] = i
            self.assertEqual(int(q), i)

    def test_RegisterFile(self):
        addr1 = TestVector(0, 3)
        addr2 = TestVector(0, 3)
        addr_w = TestVector(0, 3)
        data_w = TestVector(0, 16)
        clk = TestSignal()
        data1, data2 = RegisterFile(addr1, addr2, addr_w, data_w, clk)
        for i in xrange(2 ** 3):
            data_w[:] = i
            addr_w[:] = i
            clk.set()
            clk.reset()
        for i in xrange(2 ** 3):
            addr1[:] = i
            addr2[:] = (i + 1) % 2 ** 3
            self.assertEqual(int(data1), i)
            self.assertEqual(int(data2), (i + 1) % 2 ** 3)

    def test_AdderDelay(self):
        for bitlen in xrange(2, 65):
            a = TestVector(-1, bitlen)
            b = TestVector(-1, bitlen)
            m, m_c = simplify(*RippleCarryAdder(a, b))
            c = TestVector(-1, bitlen)
            d = TestVector(-1, bitlen)
            n, n_c = simplify(*KoggeStoneAdder(c, d))

            m_ad = calcAreaDelay(a[:] + b[:])
            print 'RippleCarryAdder:', bitlen, ':', m_ad

            n_ad = calcAreaDelay(c[:] + d[:])
            print 'KoggeStoneAdder:', bitlen, ':', n_ad
            self.assertLess(m_ad[0], n_ad[0])
            # self.assertGreaterEqual(m_ad[1], n_ad[1])


    def test_MSP430RegisterFile(self):
        clk = TestSignal()
        src_reg = TestVector(0, 4)
        dst_reg = TestVector(0, 4)
        src_incr = TestSignal()
        src_mode = TestVector(0, 2)
        dst_in = TestVector(0, 16)
        sr_in = TestVector(0, 16)
        dst_wr = TestSignal()
        bw = TestVector(0, 1)
        pc_incr = TestSignal()
        src_out, dst_out, regs, src_incr_val = MSP430.RegisterFile(pc_incr, src_reg, dst_reg, src_incr, src_mode,
                                                                   dst_in, sr_in,
                                                                   dst_wr, bw,
                                                                   clk)

        dst_wr.set(True)
        for r in xrange(16):
            if r == 3: continue  # register 3 is only constant generator
            dst_reg[:] = r
            dst_in[:] = r * r * (r + 1)
            clk.set()
            clk.reset()
            # print 'src_out', src_out
            self.assertEqual(int(src_out), 0)
            # print 'dst_out', dst_out
            self.assertEqual(int(dst_out), r * r * (r + 1))

        for r, reg in enumerate(regs):
            if r not in [2, 3]:
                # print 'reg', r, reg
                self.assertEqual(int(reg), r * r * (r + 1))

        for r in xrange(16):
            sr_in[:] = r
            clk.set()
            clk.reset()
            # print 'sr', r, regs[2]
            self.assertEqual(int(regs[2]), r)

        sr_in[:] = 3
        dst_reg[:] = 2
        dst_in[:] = 5
        clk.set()
        clk.reset()
        print 'sr', r, regs[2]
        self.assertEqual(int(regs[2]), 5)

        dst_wr.set(False)
        sr_in[:] = 4
        dst_in[:] = 6
        clk.set()
        clk.reset()
        # print 'sr', r, regs[2]
        self.assertEqual(int(regs[2]), 5)

        pc_incr.set(True)
        for r in xrange(16):
            clk.set()
            clk.reset()
            # print 'pc', regs[0]
            self.assertEqual(int(regs[0]), r * 2 + 2)

    def test_CPU_init(self):
        clk = TestSignal()
        cs = CodeSequence()

        debuglines = CPU(cs.code, clk)

        for reg in debuglines['regs']:
            print reg
            self.assertEquals(reg, Vector(0, 16))

    def test_MOV(self):
        clk = TestSignal()
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
        self.assertEquals(debuglines['regs'][4], Vector(2 ** 10, 16))
        clk.cycle(d2, CPU_state)  # MOV(N(2**6), R(4), B)
        self.assertEquals(debuglines['regs'][4], Vector(2 ** 6, 16))
        clk.cycle(d3, CPU_state)  # MOV(R(4), R(5))
        self.assertEquals(debuglines['regs'][5], Vector(2 ** 6, 16))
        clk.cycle(d4, CPU_state)  # MOV(N(2 ** 8), +N(2 ** 16 - 4))
        clk.cycle(d5, CPU_state)  # MOV(+N(2 ** 16 - 4), +N(2 ** 16 - 6))
        clk.cycle(d6, CPU_state)  # MOV(+N(2 ** 16 - 4), R(6))
        self.assertEquals(debuglines['regs'][6], Vector(2 ** 8, 16))
        clk.cycle(d7, CPU_state)  # MOV(+N(2**6 - 6), R(7))
        self.assertEquals(debuglines['regs'][7], Vector(2 ** 8, 16))

    def two_arg_alu(self, instr, op, test_seq):
        # first test just the ALU
        flags = {f: Signal() for f in MSP430.FLAGS}
        for a, b, fl, flb in test_seq:
            res, flags = MSP430.alu(MSP430.INSTRS[instr], Vector(b), Vector(a), flags, MSP430.BYTE_WORD['WORD'])
            self.assertEquals(res, Vector(op(a, b)))
            self.assertEquals(set(f for f in flags if flags[f].value), set(fl))
        flags = {f: Signal() for f in MSP430.FLAGS}
        for a, b, fl, flb in test_seq:
            res, flags = MSP430.alu(MSP430.INSTRS[instr], Vector(b, 8), Vector(a, 8), flags, MSP430.BYTE_WORD['BYTE'])
            self.assertEquals(res, Vector(op(a, b), 8))
            self.assertEquals(set(f for f in flags if flags[f].value), set(flb))

        # then test executing instructions on the alu
        clk = TestSignal()
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
            self.assertEquals(debuglines['state'], MSP430.STATES['FETCH'])
            self.assertEquals(debuglines['regs'][4], Vector(op(a, b), 16),
                              'Result wrong while testing %s %i %i: is %s, should be %s' % (instr,
                                                                                            a, b,
                                                                                            int(debuglines['regs'][4]),
                                                                                            op(a, b)))
            self.assertEquals(debuglines['regs'][2], Vector(sum(2 ** MSP430.FLAGS_SR_BITS[f] for f in fl), 16),
                              'Flags wrong while testing %s %i %i: is %s, should be %s' % (instr,
                                                                                           a, b,
                                                                                           int(debuglines['regs'][2]),
                                                                                           sum(2 **
                                                                                               MSP430.FLAGS_SR_BITS[f]
                                                                                               for f in fl)))
        for a, b, fl, flb in test_seq:
            clk.cycle(cycles.pop(0))
            self.assertEquals(debuglines['state'], MSP430.STATES['FETCH'])
            self.assertEquals(Vector(debuglines['regs'][5][:8]), Vector(op(a, b), 8),
                              'Result wrong while testing (byte) %s %i %i: is %s, should be %s' % (instr,
                                                                                                   a, b, int(
                                  Vector(debuglines['regs'][5][:8])), op(a, b)))
            self.assertEquals(debuglines['regs'][2], Vector(sum(2 ** MSP430.FLAGS_SR_BITS[f] for f in flb), 16),
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
        clk = TestSignal()
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
            self.assertEquals(debuglines['state'], MSP430.STATES['FETCH'])
            res = op(a, carry, 16)
            self.assertEquals(debuglines['regs'][4], Vector(res, 16),
                              'Result wrong while testing %s %i: is %s, should be %s' % (instr, a,
                                                                                         int(debuglines['regs'][4]),
                                                                                         res))
            self.assertEquals(debuglines['regs'][2], Vector(sum(2 ** MSP430.FLAGS_SR_BITS[f] for f in fl), 16),
                              'Flags wrong while testing %s %i: is %s, should be %s' % (instr, a,
                                                                                        int(debuglines['regs'][2]),
                                                                                        sum(2 **
                                                                                            MSP430.FLAGS_SR_BITS[f]
                                                                                            for f in fl)))
        for a, fl, flb in test_seq:
            carry = (int(debuglines['regs'][2]) >> MSP430.FLAGS_SR_BITS['c']) % 2
            clk.cycle(cycles.pop(0))
            self.assertEquals(debuglines['state'], MSP430.STATES['FETCH'])
            res = op(a, carry, 8)
            self.assertEquals(Vector(debuglines['regs'][5][:8]), Vector(res, 8),
                              'Result wrong while testing (byte) %s %i: is %s, should be %s' % (instr,
                                                                                                a, int(
                                  Vector(debuglines['regs'][5][:8])), res))
            self.assertEquals(debuglines['regs'][2], Vector(sum(2 ** MSP430.FLAGS_SR_BITS[f] for f in flb), 16),
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
        clk = TestSignal()
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
        clk.cycle(d1)
        clk.cycle(d2)
        clk.cycle(d3)
        clk.cycle(d4)
        clk.cycle(d5)
        clk.cycle(d6)
        clk.cycle(d7, debuglines)
        clk.cycle(d8, debuglines)
        clk.cycle(d9, debuglines)
        self.assertEquals(debuglines['regs'][4], Vector(v / 2, 16))
        self.assertEquals(debuglines['regs'][5], Vector(v / 4, 16))
        self.assertEquals(debuglines['regs'][6], Vector(v / 8, 16))
        self.assertEquals(debuglines['regs'][7], Vector(v / 16, 16))

    def test_PUSH(self):
        clk = TestSignal()
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
        self.assertEquals(debuglines['regs'][1], Vector(-2, 16))
        clk.cycle(d1)
        self.assertEquals(debuglines['regs'][1], Vector(-4, 16))
        clk.cycle(d2)
        self.assertEquals(debuglines['regs'][4], Vector(-2, 16))
        clk.cycle(d3)
        self.assertEquals(debuglines['regs'][1], Vector(-6, 16))
        clk.cycle(d4)
        self.assertEquals(debuglines['regs'][4], Vector(255, 16))
        clk.cycle(d5)
        clk.cycle(d6)
        self.assertEquals(debuglines['regs'][4], Vector(16676, 16))


    def test_BRANCH(self):
        clk = TestSignal()
        cs = CodeSequence()
        s0, d0 = cs.MOV(N(0xFFFF), R(10))
        s1, d1 = cs.BRANCH(R(4))

        debuglines = CPU(cs.code, clk)
        for _ in range(2):
            clk.cycle(d0)
            clk.cycle(d1)
            self.assertEquals(debuglines['regs'][0], Vector(0, 16))

    def test_JMP(self):
        clk = TestSignal()
        cs = CodeSequence()
        l1 = cs.label()
        s0, d0 = cs.MOV(N(0xFFFF), R(10))
        s1, d1 = cs.JMP(l1)

        debuglines = CPU(cs.code, clk)
        for _ in range(2):
            clk.cycle(d0)
            clk.cycle(d1)
            self.assertEquals(debuglines['regs'][0], Vector(0, 16))

    def test_JZ(self):
        clk = TestSignal()
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
            self.assertEquals(debuglines['regs'][0], Vector(0, 16))

    def test_JN(self):
        clk = TestSignal()
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
            self.assertEquals(debuglines['regs'][0], Vector(0, 16))

    def test_JC(self):
        clk = TestSignal()
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
            self.assertEquals(debuglines['regs'][0], Vector(0, 16))


    def test_CALL(self):
        clk = TestSignal()
        cs = CodeSequence()
        s0, d0 = cs.MOV(N(0xFFFF), R(1))
        s1, d1 = cs.MOV(~R(1), R(4))
        s2, d2 = cs.CALL(N(s0 * 2))

        debuglines = CPU(cs.code, clk)
        clk.cycle(d0)
        self.assertEquals(debuglines['regs'][1], Vector(-2, 16))
        clk.cycle(d1)
        self.assertEquals(debuglines['regs'][4], Vector(0, 16))
        clk.cycle(d2)
        self.assertEquals(debuglines['regs'][0], Vector(s0 * 2, 16))
        self.assertEquals(debuglines['regs'][1], Vector(-4, 16))
        clk.cycle(d1)
        self.assertEquals(debuglines['regs'][4], Vector((s0 + s1 + s2) * 2, 16))

    def test_alu(self):
        res = MSP430.alu(MSP430.INSTRS['SUB'], Vector(191), Vector(209), {'c': Signal()}, MSP430.BYTE_WORD['WORD'])
        print res

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
