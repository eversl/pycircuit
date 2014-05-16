'''
Created on Feb 6, 2014

@author: matrix1
'''
import unittest
from PyCircuit import TestSignal, FullAdder, SRLatch, DLatch, DFlipFlop, \
    intToSignals, signalsToInt, RippleCarryAdder, Negate, Vector, Multiplier, \
    Decoder, Memory, RegisterFile, calcAreaDelay, KoggeStoneAdder,\
    TestVector
from MSP430 import CPU, CodeSequence, N, R, B
import MSP430


class Test(unittest.TestCase):
    def test_str(self):
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
                als = intToSignals(a, 16)
                bls = intToSignals(b, 16)
                sls, c_out = RippleCarryAdder(als, bls)
                sum = signalsToInt(sls, True)
                self.assertEqual(sum, a + b)
                self.assertEqual(c_out.value, a < 0)

    def _test_KoggeStoneAdder(self):
        for a in xrange(-256, 255, 67):
            for b in xrange(-22756, 32767, 1453):
                als = intToSignals(a, 16)
                bls = intToSignals(b, 16)
                sls, c_out = KoggeStoneAdder(als, bls)
                sum = signalsToInt(sls, True)
                self.assertEqual(sum, a + b)
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
                als = intToSignals(num, 16)
                sls = Negate(als)
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


    def _test_Memory(self):
        a = TestVector(0, 8)
        d = TestVector(0, 16)
        mem_wr = TestVector(0,1)
        q = Memory(a, d, mem_wr)
        print calcAreaDelay(a)
        print calcAreaDelay(d)
        print calcAreaDelay(mem_wr[:])
        for i in xrange(2 ** 8):
            d[:] = i
            a[:] = i
            mem_wr[0].set()
            mem_wr[0].reset()
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
            addr2[:] = (i + 1) % 2**3
            self.assertEqual(int(data1), i)
            self.assertEqual(int(data2), (i + 1) % 2**3)
            
    def _test_AdderDelay(self):
        for bitlen in xrange(2, 65):
            a = Vector(-1, bitlen)
            b = Vector(-1, bitlen)
            m,m_c = RippleCarryAdder(a, b)
            c = Vector(-1, bitlen)
            d = Vector(-1, bitlen)
            n,n_c = KoggeStoneAdder(c, d)
                    
            m_ad = calcAreaDelay(a[:] + b[:])
            print 'RippleCarryAdder:', bitlen, ':', m_ad
            
            n_ad = calcAreaDelay(c[:] + d[:])
            print 'KoggeStoneAdder:', bitlen, ':', n_ad
            self.assertLess(m_ad[0], n_ad[0]) 
            self.assertGreater(m_ad[1], n_ad[1]) 
            
    
    def test_MSP430RegisterFile(self):
        clk = TestSignal()
        reset = TestSignal()
        src_reg = Vector(0,4)
        dst_reg = TestVector(0,4)
        src_incr = TestVector(0,1)
        dst_in = TestVector(0, 16)
        dst_wr = TestSignal(True)
        bw = TestSignal(0)
        pc_incr = TestSignal()
        src_out, dst_out, _ = MSP430.RegisterFile(pc_incr, src_reg, dst_reg, src_incr, dst_in, dst_wr, bw, clk, reset)
        
        print 'src_out', src_out
        print 'dst_out', dst_out
        for r in xrange(16):
            dst_reg[:] = r
            dst_in[:] = r + 1
            clk.set()
            clk.reset()
            print 'src_out', src_out
            print 'dst_out', dst_out

    def test_MSP430(self):
        clk = TestSignal()
        code_sequence = CodeSequence()
        code_sequence.MOV(~R(0), R(8), B)
        code_sequence.ADD(N(5000), R(4))
        code_sequence.ADDC(R(4), R(5)+20)
        code_sequence.BIS(R(5)+20, R(6))

        debuglines = CPU(code_sequence.code, clk)
        
        for _ in xrange(10):
            print '-' * 30
            for k in debuglines:
                print k, ':', debuglines[k]
            clk.set()
            clk.reset()
        self.fail('Debug')
            
if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
