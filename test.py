'''
Created on Feb 6, 2014

@author: matrix1
'''
import unittest
from PyCircuit import TestSignal, FullAdder, SRLatch, DLatch, DFlipFlop, \
    intToSignals, signalsToInt, RippleCarryAdder, Negate, Vector, Multiplier, \
    Decoder, Memory, RegisterFile, calcAreaDelay, KoggeStoneAdder,\
    TestVector, Signal, GuardedMemory, sim_steps, GuardedVector
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


    def test_Memory(self):
        a = TestVector(0, 8)
        d = TestVector(0, 16)
        mem_wr = TestSignal(0)
        q = GuardedVector(GuardedMemory(a, d, mem_wr))
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
        src_reg = TestVector(0,4)
        dst_reg = TestVector(0,4)
        src_incr = TestSignal()
        dst_in = TestVector(0, 16)
        sr_in = TestVector(0, 16)
        dst_wr = TestSignal()
        bw = TestSignal(0)
        pc_incr = TestSignal()
        src_out, dst_out, regs = MSP430.RegisterFile(pc_incr, src_reg, dst_reg, src_incr, dst_in, sr_in, dst_wr, bw, clk, reset)
        
        dst_wr.set(True)
        for r in xrange(16):
            dst_reg[:] = r
            dst_in[:] = r * r * 4
            clk.set()
            clk.reset()
            # print 'src_out', src_out
            self.assertEqual(int(src_out), 0)
            # print 'dst_out', dst_out
            self.assertEqual(int(dst_out), r * r * 4)

        for r, reg in enumerate(regs):
            if r not in [2, 3]:
                # print 'reg', r, reg
                self.assertEqual(int(reg), r * r * 4)

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
        code_sequence = CodeSequence()

        debuglines = CPU(code_sequence.code, clk)

        for reg in debuglines['regs']:
            print reg
            self.assertEquals(reg, Vector(0, 16))

    def test_MOV(self):
        clk = TestSignal()
        code_sequence = CodeSequence()
        code_sequence.MOV(N(2**10), R(4))
        code_sequence.MOV(N(2**10 + 2**6), R(4), B)
        code_sequence.MOV(R(4), R(5))
        code_sequence.MOV(N(2**8), +N(2**16 - 4))
        code_sequence.MOV(R(5), R(2))
        code_sequence.MOV(+N(2**16 - 4), +N(2**16 - 6))
        code_sequence.MOV(+N(2**16 - 4), R(6))
        code_sequence.MOV(+N(2**16 - 6), R(7))

        debuglines = CPU(code_sequence.code, clk)

        CPU_state = debuglines['state']
        clk.cycle(2, CPU_state)  # MOV(N(2**10), R(4))
        self.assertEquals(debuglines['regs'][4], Vector(2**10, 16))
        clk.cycle(2, CPU_state)  # MOV(N(2**6), R(4), B)
        self.assertEquals(debuglines['regs'][4], Vector(2**6, 16))
        clk.cycle(1, CPU_state)  # MOV(R(4), R(5))
        self.assertEquals(debuglines['regs'][5], Vector(2**6, 16))
        clk.cycle(6, debuglines)  # MOV(N(2**8), +N(2**7 - 4)), MOV(R(5), R(2))
        self.assertEquals(debuglines['regs'][2], Vector(2**6, 16))
        clk.cycle(9, debuglines)  # MOV(+N(2**6 - 4), +N(2**6 - 6)), MOV(+N(2**7 - 4), R(6))
        self.assertEquals(debuglines['regs'][6], Vector(2**8, 16))
        clk.cycle(3, CPU_state) # MOV(+N(2**6 - 6), R(7))
        self.assertEquals(debuglines['regs'][7], Vector(2**8, 16))

    def two_arg_alu(self, instr, op, test_seq):
        clk = TestSignal()
        code_sequence = CodeSequence()
        for a, b, fl in test_seq:
            code_sequence.MOV(N(a), R(4))
            code_sequence.MOV(N(a), R(5), B)
            fn = getattr(code_sequence, instr)
            fn(N(b), R(4))
            fn(N(b), R(5), B)
        debuglines = CPU(code_sequence.code, clk)
        for a, b, fl in test_seq:
            clk.cycle(4)

            for k in debuglines:
                print k, ":", debuglines[k]
            self.assertEquals(debuglines['regs'][4], Vector(op(a, b), 16),
                              'Result wrong while testing %s %i %i: is %s, should be %s' % (instr,
                              a, b, int(debuglines['regs'][4]), op(a, b)))
            self.assertEquals(debuglines['regs'][2], Vector(sum(2 ** MSP430.FLAGS_SR_BITS[f] for f in fl), 16),
                              'Flags wrong while testing %s %i %i: is %s, should be %s' % (instr,
                              a, b, int(debuglines['regs'][2]), sum(2 ** MSP430.FLAGS_SR_BITS[f] for f in fl)))
            clk.cycle(4)
            self.assertEquals(debuglines['regs'][5], Vector(op(a, b), 16),
                              'Result wrong while testing %s %i %i: is %s, should be %s' % (instr,
                              a, b, int(debuglines['regs'][5]), op(a, b)))
            self.assertEquals(debuglines['regs'][2], Vector(sum(2 ** MSP430.FLAGS_SR_BITS[f] for f in fl), 16),
                              'Flags wrong while testing %s %i %i: is %s, should be %s' % (instr,
                              a, b, int(debuglines['regs'][2]), sum(2 ** MSP430.FLAGS_SR_BITS[f] for f in fl)))

    def test_ADD(self):
        test_seq = [(1, 2, ''), (-1, 2, 'c'), (1, -2, 'n'), (2, -2, 'cz'), (3857, -4297, 'n'), (-2**15, 2**15, 'vcz'),
                    (2**15 - 2, 2**15 - 1, 'vn'), (-2**15 - 2, -2**15 + 1, 'n')]
        self.two_arg_alu('ADD', lambda a, b: a + b, test_seq)

    def test_ADDC(self):
        test_seq = [(1, 2, ''), (-1, 2, 'c'), (4, 3, ''), (2, -2, 'cz'), (3857, -4297, 'n'), (-2**15, 2**15, 'vcz'),
                    (2**15 - 1, 2**15 - 1, 'vn'), (-2**15 - 3, -2**15 + 2, 'n')]
        self.two_arg_alu('ADDC', lambda a, b: a + b + (b % 2), test_seq)

    def test_SUB(self):
        test_seq = [(1, 2, 'c'), (-1, 3, ''), (1, -2, 'nc'), (2, -2, 'nc'), (3857, -4297, 'nc'), (-2**15, 2**15, 'vcz'),
                    (2**15 - 2, 2**15 - 1, 'c'), (-2**15 - 2, -2**15 + 1, 'vc')]
        self.two_arg_alu('SUB', lambda a, b: b - a, test_seq)

    def test_SUBC(self):
        test_seq = [(1, 2, 'c'), (-1, 3, ''), (1, -2, 'nc'), (3, -3, 'nc'), (3857, -4297, 'nc'), (-2**15 - 2, 2**15 + 1, 'vc'),
                    (2**15 - 2, 2**15 - 1, 'c'), (-2**15 - 2, -2**15 + 1, 'vc')]
        self.two_arg_alu('SUBC', lambda a, b: b - a + (b % 2), test_seq)

    def test_CMP(self):
        test_seq = [(1, 2, 'c'), (-1, 3, ''), (1, -2, 'nc'), (2, -2, 'nc'), (3857, -4297, 'nc'), (-2**15, 2**15, 'vcz'),
                    (2**15 - 2, 2**15 - 1, 'c'), (-2**15 - 2, -2**15 + 1, 'vc')]
        self.two_arg_alu('CMP', lambda a, b: a, test_seq)

    def test_AND(self):
        test_seq = [(1, 2, 'z'), (-1, 3, 'c'), (1, -2, 'z'), (2, -2, 'c'), (3857, -4297, 'c'), (-2**15, 2**15, 'nc'),
                    (2**15 - 2, 2**15 - 1, 'c'), (-2**15 - 2, -2**15 + 1, 'z')]
        self.two_arg_alu('AND', lambda a, b: a & b, test_seq)

    def test_AND(self):
        test_seq = [(1, 2, 'z'), (-1, 3, 'c'), (1, -2, 'z'), (2, -2, 'c'), (3857, -4297, 'c'), (-2**15, 2**15, 'nc'),
                    (2**15 - 2, 2**15 - 1, 'c'), (-2**15 - 2, -2**15 + 1, 'z')]
        self.two_arg_alu('AND', lambda a, b: a & b, test_seq)



    @unittest.skip("debugging")
    def test_MSP430(self):
        clk = TestSignal()
        code_sequence = CodeSequence()
        code_sequence.MOV(N(10), R(4))
        code_sequence.SUB(N(3), R(4))
        code_sequence.MOV(N(7), R(5))
        code_sequence.BIS(N(132), R(5))

        debuglines = CPU(code_sequence.code, clk)
        
        for _ in xrange(10):
            print '-' * 30
            for k in sorted(debuglines.keys()):
                print k, ':', debuglines[k]
            clk.set()
            clk.reset()
        self.fail('Debug')

    def test_add(self):
        r, c = KoggeStoneAdder(Vector(1, 4), Vector(2, 4), Signal(1))
        print Vector(r)
            
if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
