'''
Created on Feb 6, 2014

@author: matrix1
'''
import unittest

from PyCircuit import TestSignal, FullAdder, SRLatch, DLatch, DFlipFlop, \
    intToSignals, signalsToInt, RippleCarryAdder, Vector, Multiplier, \
    Decoder, Memory, ROM, calcAreaDelay, KoggeStoneAdder, \
    TestVector, DecimalAdder, simplify, Case, DontCare, Clock, Register, ConstantSignal, ConstantVector, \
    VFalse, monitor, DontCareVal


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
        for i in range(-100, 256):
            ls = intToSignals(i, 9)
            j = signalsToInt(ls, True)
            self.assertEqual(i, j)

    def test_RippleCarryAdder(self):
        for a in range(-256, 255, 67):
            for b in range(-22756, 32767, 1453):
                for c in range(2):
                    # current_cache.clear()
                    als = ConstantVector(a, 16)
                    bls = ConstantVector(b, 16)
                    sls, c_out = RippleCarryAdder(als, bls, bool(c))
                    self.assertVectorEqual(sls, a + b + c)
                    self.assertVectorEqual(c_out, a < 0)

    def test_KoggeStoneAdder(self):
        for a in range(-256, 255, 67):
            for b in range(-22756, 32767, 1453):
                for c in range(2):
                    als = ConstantVector(a, 16)
                    bls = ConstantVector(b, 16)
                    sls, c_out = KoggeStoneAdder(als, bls, bool(c))
                    self.assertCurrent(sls)
                    self.assertVectorEqual(sls, a + b + c)
                    self.assertEqual(bool(c_out), a < 0)

    def test_add(self):
        als = TestVector(0, 16)
        bls = TestVector(0, 16)
        sls = als + bls
        for a in range(-256, 255, 67):
            for b in range(-22756, 32767, 1453):
                als[:] = a
                bls[:] = b
                self.assertCurrent(sls)
                self.assertVectorEqual(sls, a + b)

    def assertCurrent(self, vec):
        self.assertEqual(vec.current(), signalsToInt(vec.ls, False))

    def test_DecimalAdder(self):
        for a in range(0, 10000, 907):
            d_a = Vector.concat(*[ConstantVector(int(d), 4) for d in reversed("%04u" % a)])
            for b in range(0, 10000, 359):
                d_b = Vector.concat(*[ConstantVector(int(d), 4) for d in reversed("%04u" % b)])
                for c in range(2):
                    sls, c_out = DecimalAdder(d_a, d_b, bool(c))
                    self.assertCurrent(sls)
                    sum = int("%x" % sls.concat(c_out).toUint())
                    self.assertEqual(sum, a + b + c, "%i != %i (%i + %i + %i)" % (sum, a + b + c, a, b, c))

    def test_SimplifyConst(self):
        for a in range(-256, 255, 67):
            for b in range(-22756, 32767, 1453):
                als = ConstantVector(a, 16)
                bls = ConstantVector(b, 16)
                sls, c_out = KoggeStoneAdder(als, bls)
                self.assertCurrent(sls)
                sls, c_out = simplify(sls, c_out)
                self.assertTrue(all(isinstance(s, ConstantSignal) for s in sls.ls))
                self.assertCurrent(sls)

                self.assertVectorEqual(sls, a + b)
                self.assertVectorEqual(c_out.ls[0].value, a < 0)

    def _test_Simplify(self):
        for a in range(-256, 255, 67):
            als = ConstantVector(a, 16)
            bls = TestVector(0, 16)
            cs = TestVector(0, 1)
            sls, c_out = KoggeStoneAdder(als, bls, cs)
            n_ad = calcAreaDelay(als.concat(bls, cs))
            print(a, 'Before:', ':', n_ad, end=' ')

            sls, c_out = simplify(sls, c_out)

            n_ad = calcAreaDelay(als.concat(bls, cs))
            print('After: :', n_ad)

            for b in range(-22756, 32767, 1453):
                for c in range(2):
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
        for num in range(-22756, 32767, 1453):
            als = ConstantVector(num, 16)
            sls = -als
            self.assertVectorEqual(sls, -num)

    def test_Multiplier(self):
        bitlen = 8
        a = TestVector(255, bitlen)
        b = TestVector(255, bitlen)
        m = Multiplier(a, b, False)
        # noinspection PyCallByClass
        print(calcAreaDelay(Vector.concat(a, b)))
        self.assertEqual(len(m), bitlen * 2)
        for av in range(2, 2 ** bitlen, 23):
            a[:] = av
            for bv in range(3, 2 ** bitlen, 27):
                b[:] = bv
                self.assertVectorEqual(m, av * bv)

    def test_Decoder(self):
        a = TestVector(0, 8)
        d = Decoder(a)
        for i in range(2 ** 8):
            a[:] = i
            self.assertVectorEqual(d, 2 ** i)

    def test_Memory(self):
        clk = Clock()
        SZ = 8
        a = TestVector(0, SZ)
        d = TestVector(0, 16)
        mem_wr = TestVector(0, 1)
        q = Memory(clk, a, d, mem_wr)
        self.assertVectorEqual(q, d)
        self.assertCurrent(q)
        print(calcAreaDelay(a.concat(d).concat(mem_wr)))
        for i in range(2 ** SZ):
            a[:] = i
            self.assertVectorEqual(q, 0)
            self.assertCurrent(q)
        for i in range(2 ** SZ):
            d[:] = i
            a[:] = i
            mem_wr[:] = 1
            self.assertVectorEqual(q, d)
            self.assertCurrent(q)
            mem_wr[:] = 0
            self.assertVectorEqual(q, d)
            self.assertCurrent(q)
        for i in range(2 ** SZ):
            a[:] = i
            self.assertVectorEqual(q, i)
            self.assertCurrent(q)

    def test_ROM(self):
        clk = Clock()
        size = 10
        a = TestVector(0, size)
        d = TestVector(0, 16)
        mem_wr = VFalse
        q = ROM(clk, a, d, mem_wr, list(range(2 ** size)))
        self.assertCurrent(q)
        for i in range(2 ** size):
            a[:] = i
            self.assertVectorEqual(q, i)
            self.assertCurrent(q)

    def _test_AdderDelay(self):
        for bitlen in range(2, 65):
            a = TestVector(-1, bitlen)
            b = TestVector(-1, bitlen)
            m, m_c = simplify(*RippleCarryAdder(a.ls, b.ls))
            c = TestVector(-1, bitlen)
            d = TestVector(-1, bitlen)
            n, n_c = simplify(*KoggeStoneAdder(c, d))

            m_ad = calcAreaDelay(a.concat(b))
            print('RippleCarryAdder:', bitlen, ':', m_ad)

            n_ad = calcAreaDelay(c.concat(d))
            print('KoggeStoneAdder:', bitlen, ':', n_ad)
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
        self.assertTrue(all(type(el.value) is DontCare for el in res.ls))
        self.assertCurrent(res)

    def cycleAndCheck(self, clk, dst_out, regs, src_incr_val, src_out):
        clk.cycle()
        self.assertCurrent(src_out)
        self.assertCurrent(dst_out)
        for reg in regs:
            self.assertCurrent(reg)
        self.assertCurrent(src_incr_val)

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

    def test_DontCareDecoder(self):
        a = TestVector(0, 4)
        monitor(a)
        d = Decoder(a)
        monitor(d.ls)
        for m in range(2 ** 4):
            for i in range(2 ** 4):
                a[:] = DontCareVal(i, m)
                # self.assertVectorEqual(d, 2 ** i)
                self.assertCurrent(d)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
