'''
Created on Feb 6, 2014

@author: matrix1
'''
import unittest

import MSP430
from MSP430 import CPU, CodeSequence, N, R, B, R_PC, R_SP, R_SR
from PyCircuit import signalsToInt, Vector, Decoder, calcAreaDelay, TestVector, simplify, Clock, ConstantVector, \
    monitor, DontCareVal


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

    def assertCurrent(self, vec):
        self.assertEqual(vec.current(), signalsToInt(vec.ls, False))

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

        for r in range(16):
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

        for r in range(16):
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
        for r in range(16):
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
            print(reg)
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
                                                                                                 int(debuglines['regs'][
                                                                                                         4]),
                                                                                                 op(a, b)))
            self.assertVectorEqual(debuglines['regs'][2], sum(2 ** MSP430.FLAGS_SR_BITS[f] for f in fl),
                                   'Flags wrong while testing %s %i %i: is %s, should be %s' % (instr,
                                                                                                a, b,
                                                                                                int(debuglines['regs'][
                                                                                                        2]),
                                                                                                sum(2 **
                                                                                                    MSP430.FLAGS_SR_BITS[
                                                                                                        f]
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
                                                                                              int(debuglines['regs'][
                                                                                                      4]),
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

    def _test_SimplifyCPU(self):
        clk = Clock()
        cs = CodeSequence()
        debuglines = CPU(cs.code, clk)
        print(debuglines)
        mem_out = debuglines['mem_out']
        before = calcAreaDelay(mem_out.ls)
        from sys import setrecursionlimit
        setrecursionlimit(int(1e3))
        simplify(mem_out)
        print(before, calcAreaDelay(mem_out.ls))

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
