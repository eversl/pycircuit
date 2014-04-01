'''
Created on Feb 6, 2014

@author: matrix1
'''
import unittest
from PyCircuit import Signal, FullAdder, SRLatch, DLatch, DFlipFlop, Not, \
    intToSignals, signalsToInt, RippleCarryAdder, Negate, Vector, Multiplier, \
    Decoder, Memory, RegisterFile, calcAreaDelay, KoggeStoneAdder
from MSP430 import CPU


class Test(unittest.TestCase):
    def test_str(self):
        self.assertEqual(str(Signal(0)), 'Signal(False)')
        self.assertEqual(str(Signal(1)), 'Signal(True)') 
        self.assertEqual(str(Vector(-10)), 'Vector(-10, 16)') 

    def test_SRLatch(self):    
        r = Signal()
        s = Signal()
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
        d = Signal()
        e = Signal(True)    
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
        d = Signal()
        clk = Signal()    
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
        clk.reset()
        self.assertEquals([sig.value for sig in signals], [False, True])
 
    def test_FullAdder(self):
        a, b, c = Signal(), Signal(), Signal()
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

    def test_KoggeStoneAdder(self):
        for a in xrange(-256, 255, 67):
            for b in xrange(-22756, 32767, 1453):
                als = intToSignals(a, 16)
                bls = intToSignals(b, 16)
                sls, c_out = KoggeStoneAdder(als, bls)
                sum = signalsToInt(sls, True)
                self.assertEqual(sum, a + b)
                self.assertEqual(c_out.value, a < 0)

    def test_arith(self):
        self.assertEquals(int(-Vector(10)), -10)
        self.assertEquals(int(Vector(10) + Vector(-24)), 10 + -24)
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
        a = Vector(0, bitlen)
        b = Vector(0, bitlen)
        m = Vector(Multiplier(a, b, False))
        print calcAreaDelay(a[:] + b[:])
        self.assertEqual(len(m), bitlen * 2)
        for av in xrange(2, 2 ** bitlen, 23):
            a[:] = av
            for bv in xrange(3, 2 ** bitlen, 27):
                b[:] = bv
                self.assertEqual(signalsToInt(m, False), av * bv)
    

    def test_Decoder(self):
        a = Vector(0, 8)
        d = Decoder(a)
        for i in xrange(2 ** 8):
            a[:] = i
            self.assertEqual(signalsToInt(Vector(d), False), 2 ** i)


    def test_Memory(self):
        a = Vector(0, 8)
        d = Vector(0, 16)
        mem_wr = Signal()
        q = Memory(a, d, mem_wr)
        print calcAreaDelay(a)
        print calcAreaDelay(d)
        print calcAreaDelay([mem_wr])
        for i in xrange(2 ** 8):
            d[:] = i
            a[:] = i
            mem_wr.set()
            mem_wr.reset()
        for i in xrange(2 ** 8):
            a[:] = i
            self.assertEqual(int(q), i)

    def test_RegisterFile(self):
        addr1 = Vector(0, 3)
        addr2 = Vector(0, 3)
        addr_w = Vector(0, 3)
        data_w = Vector(0, 16)
        clk = Signal()
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
            
    def test_AdderDelay(self):
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
            
    
    def test_CPU(self):
        clk = Signal()
        debuglines = CPU(clk)
        
        for _ in xrange(10):
            clk.set()
            clk.reset()
            for k in debuglines:
                print k, ':', debuglines[k]
        self.fail('Debug')
            
if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
