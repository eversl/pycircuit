'''
Created on Mar 21, 2014

@author: leonevers
'''
from PyCircuit import Vector, Signal, Memory, Decoder, FlipFlops, FeedbackVector, If, zip_all, FeedbackSignal, Or, \
    calcAreaDelay, \
    Enum, Case, EnumVector, KoggeStoneAdder, Negate, And, signalsToInt, DecimalAdder, TrueInCase


def RegisterFile(pc_incr, src_reg, dst_reg, src_incr, src_mode, dst_in, sr_in, dst_wr, bw, clk, reset):
    dst_lines = Decoder(dst_reg)
    src_lines = Decoder(src_reg)

    src_incr_lines = FeedbackVector(0, len(src_lines))
    dst_wr_lines = FeedbackVector(0, len(dst_lines))
    dst_wr_val = FeedbackVector(0, 16)
    src_incr_val = FeedbackVector(0, 16)
    old_outs = [FeedbackVector(0, 16) for _ in dst_lines]

    q_outs = [Vector(If(reset, Vector(0, 16),
                        If(src_incr_line, src_incr_val,
                           If(dst_wr_line, dst_wr_val,
                              old_out))))
              for src_incr_line, dst_wr_line, old_out in zip_all(src_incr_lines, dst_wr_lines, old_outs)]
    q_outs[0] = Vector([Signal(0)] + q_outs[0][1:])
    q_outs[1] = Vector([Signal(0)] + q_outs[1][1:])

    src_out = Vector([Or(*l) for l in zip(*(src_line & (Case(src_mode, {SRC_M['M_INDEX']: Vector(0, 16),
                                                                        SRC_M['M_INDIRECT']: Vector(4, 16),
                                                                        SRC_M['M_INCR']: Vector(8, 16)},
                                                             q_out) if q_out is q_outs[2]
                                                        else Case(src_mode, {SRC_M['M_INDEX']: Vector(1, 16),
                                                                             SRC_M['M_INDIRECT']: Vector(2, 16),
                                                                             SRC_M['M_INCR']: Vector(-1, 16)},
                                                                  q_out) if q_out is q_outs[3]
    else q_out) for src_line, q_out in zip_all(src_lines, q_outs)))])
    dst_out = Vector([Or(*l) for l in zip(*(dst_line & q_out for dst_line, q_out in zip_all(dst_lines, q_outs)))])

    for out, old_out in zip_all(q_outs, old_outs):
        flip_flops = FlipFlops(
            If(pc_incr, out + Vector(2, 16), out) if out is q_outs[0] else
            If(dst_wr & ~dst_lines[2], sr_in, out) if out is q_outs[2] else out, clk)

        old_out.connect(Vector([Signal(0)] + flip_flops[1:]) if out is q_outs[0] or out is q_outs[1] else
                        Vector(0, 16) if out is q_outs[3] else flip_flops)

    incr_op = If(~ bw[0] | src_lines[0] | src_lines[1], Vector(2, 16), Vector(1, 16))

    src_incr_val.connect(FlipFlops(src_out + incr_op, clk))
    src_incr_lines.connect(FlipFlops((src_incr & src_line for src_line in src_lines), clk))
    dst_wr_lines.connect(FlipFlops((dst_wr & dst_line for dst_line in dst_lines), clk))
    dst_wr_val.connect(FlipFlops(dst_in, clk))
    return src_out, dst_out, q_outs, src_incr_val


I_TYPES = Enum('IT_TWO', 'IT_NONE', 'IT_ONE', 'IT_JUMP')
IT_TWO_INS = Enum('_0', '_1', '_2', '_3', )
BYTE_WORD = Enum('WORD', 'BYTE')
FLAGS_SR_BITS = {'c': 0, 'z': 1, 'n': 2, 'v': 8}
FLAGS = 'cznv'


def decodeInstr(word):
    inst_type = EnumVector(I_TYPES, If(word[15] | (~word[15] & word[14]), I_TYPES['IT_TWO'],
                                       If(word[13], I_TYPES['IT_JUMP'], I_TYPES['IT_ONE'])))
    instr = EnumVector(INSTRS, If(word[15] | (~word[15] & word[14]), Vector(word[12:] + [Signal()]),
                                  If(word[13], Vector(word[10:13] + I_TYPES['IT_JUMP'][:]),
                                     Vector(word[7:10] + I_TYPES['IT_ONE'][:]))))
    dst_reg = Case(inst_type, {I_TYPES['IT_JUMP']: Vector(0, 4)},
                   Vector(word[0:4]))
    src_reg = Case(inst_type, {I_TYPES['IT_TWO']: Vector(word[8:12]),
                               I_TYPES['IT_ONE']: dst_reg})
    bw = Case(inst_type, {I_TYPES['IT_JUMP']: BYTE_WORD['WORD']},
              EnumVector(BYTE_WORD, word[6]))
    a_s = EnumVector(SRC_M, word[4:6])
    a_d = EnumVector(DST_M, word[7])
    offset = Vector(word[:10])

    return instr, inst_type, src_reg, dst_reg, bw, a_s, a_d, offset


M_DIRECT = 0
M_INDEX = 1
M_INDIRECT = 2
M_INCR = 3

SRC_M = Enum('M_DIRECT', 'M_INDEX', 'M_INDIRECT', 'M_INCR')
DST_M = Enum('M_DIRECT', 'M_INDEX')


class R(object):
    # register mode  (R(R_XX))
    def __init__(self, reg):
        self.reg = reg
        self.mode = M_DIRECT

    # indirect autoincrement mode  (+R(R_XX))
    def __pos__(self):
        assert (self.mode == M_DIRECT)
        self.mode = M_INCR
        return self

    # Indirect register mode  (~R(R_XX))
    def __invert__(self):
        assert (self.mode == M_DIRECT)
        self.mode = M_INDIRECT
        return self

    # indexed mode  (R(R_XX) + 10)
    def __add__(self, num):
        assert (self.mode == M_DIRECT)
        self.mode = M_INDEX
        self.index = num
        return self


R_PC = 0
R_SP = 1
R_SR = 2
R_CG = 3

M_IMM = 4
M_SYM = 5
M_ABS = 6


class N(R):
    # Immediate mode  (N(10))
    def __init__(self, num):
        super(N, self).__init__(R_PC)
        self.index = num
        self.n_mode = M_IMM
        self.mode = M_INCR

    # Symbolic mode  (~N(10))
    def __invert__(self):
        assert (self.n_mode == M_IMM)
        self.n_mode = M_SYM
        self.mode = M_INDEX
        self.reg = R_PC
        return self

    # Absolute mode  (+N(10))
    def __pos__(self):
        assert (self.n_mode == M_IMM)
        self.n_mode = M_ABS
        self.mode = M_INDEX
        self.reg = R_SR
        return self


TWO_INSTR = {'MOV': 0b0100,
             'ADD': 0b0101,
             'ADDC': 0b0110,
             'SUBC': 0b0111,
             'SUB': 0b1000,
             'CMP': 0b1001,
             'DADD': 0b1010,
             'BIT': 0b1011,
             'BIC': 0b1100,
             'BIS': 0b1101,
             'XOR': 0b1110,
             'AND': 0b1111}

ONE_INSTR = {'RRC': 0b000,
             'SWPB': 0b001,
             'RRA': 0b010,
             'SXT': 0b011,
             'PUSH': 0b100,
             'CALL': 0b101,
             'RETI': 0b110}

JMP_INSTR = {'JNZ': 0b000,
             'JZ': 0b001,
             'JNC': 0b010,
             'JC': 0b011,
             'JN': 0b100,
             'JGE': 0b101,
             'JL': 0b110,
             'JMP': 0b111}

INSTRS = Enum(dict(TWO_INSTR.items() +
                   [(i, ONE_INSTR[i] + (signalsToInt(I_TYPES['IT_ONE'], False) << 3)) for i in ONE_INSTR] +
                   [(i, JMP_INSTR[i] + (signalsToInt(I_TYPES['IT_JUMP'], False) << 3)) for i in JMP_INSTR]))
B = True


class CodeSequence(object):
    def __init__(self):
        self.code = []
        self.labels = []

    def __getattr__(self, name):
        if name in TWO_INSTR:
            def fn(src, dst, b=False):
                return self.two_operand_instr(TWO_INSTR[name], src, dst, b)

            return fn

        elif name in ONE_INSTR:
            def fn(dst, b=False):
                return self.one_operand_instr(ONE_INSTR[name], dst, b)

            return fn

        elif name in JMP_INSTR:
            def fn(label):
                return self.jmp_instr(JMP_INSTR[name], label)

            return fn
        else:
            raise AttributeError


    # =============  emulated instructions ===================

    # Add carry to destination
    def ADC(self, dst, b):
        return self.ADDC(N(0), dst, b)

    # Branch to destination
    def BRANCH(self, target):
        return self.MOV(target, R(0))

    # Clear destination
    def CLR(self, dst, b):
        return self.MOV(N(0), dst, b)

    # Clear carry bit
    def CLRC(self):
        return self.BIC(N(1 << FLAGS_SR_BITS['c']), R(2))

    # Clear negative bit
    def CLRN(self):
        return self.BIC(N(1 << FLAGS_SR_BITS['n']), R(2))

    # Clear zero bit
    def CLRZ(self):
        return self.BIC(N(1 << FLAGS_SR_BITS['z']), R(2))

    # Add carry decimally to destination
    def DADC(self, dst, b):
        return self.ADDC(N(0), dst, b)

    # Decrement destination
    def DEC(self, dst, b):
        return self.SUB(N(1), dst, b)

    # Double-decrement destination
    def DECD(self, dst, b):
        return self.SUB(N(2), dst, b)

    # Disable (general) interrupts
    def DINT(self):
        return self.BIC(N(8), R(2))

    # Enable (general) interrupts
    def EINT(self):
        return self.BIS(N(8), R(2))

    # Increment destination
    def INC(self, dst, b):
        return self.ADD(N(1), dst, b)

    # Double-increment destination
    def INCD(self, dst, b):
        return self.ADD(N(2), dst, b)

    # Invert destination
    def INV(self, dst, b):
        return self.XOR(N(-1), dst, b)

    # Jump if higher or same
    def JHS(self, target):
        return self.JC(target)

    # Jump if equal
    def JEQ(self, target):
        return self.JZ(target)

    # Jump if lower
    def JLO(self, target):
        return self.JNC(target)

    # Jump if not equal
    def JNE(self, target):
        return self.JNZ(target)

    # No operation
    def NOP(self):
        return self.MOV(N(0), R(3))

    # Pop byte/word from stack to destination
    def POP(self, dst, b):
        return self.MOV(+R(1), dst, b)

    # Return from subroutine
    def RET(self):
        return self.MOV(+R(1), R(0))

    # Rotate left arithmetically
    def RLA(self, dst, b):
        return self.ADD(dst, dst, b)

    # Rotate left through carry
    def RLA(self, dst, b):
        return self.ADDC(dst, dst, b)

    # Rotate left through carry
    def SBC(self, dst, b):
        return self.SUBC(N(0), dst, b)

    # Set carry bit
    def SETC(self):
        return self.BIS(N(1 << FLAGS_SR_BITS['c']), R(2))

    # Set negative bit
    def SETN(self):
        return self.BIS(N(1 << FLAGS_SR_BITS['n']), R(2))

    # Set zero bit
    def SETZ(self):
        return self.BIS(N(1 << FLAGS_SR_BITS['z']), R(2))

    # Test destination
    def TST(self, dst, b):
        return self.CMP(N(0), dst, b)



    def two_operand_instr(self, instr, src, dst, b=False):
        old_len = len(self.code)
        assert src.mode in [M_DIRECT, M_INDEX, M_INDIRECT, M_INCR]
        assert dst.mode in [M_DIRECT, M_INDEX]

        special_constant = False
        if src.mode == M_INCR and src.reg == R_PC and src.index in [-1, 0, 1, 2, 4, 8]:
            special_constant = True
            src.reg, src.mode = {-1: (3, 0b11),
                                 0: (3, 0b00),
                                 1: (3, 0b01),
                                 2: (3, 0b10),
                                 4: (2, 0b10),
                                 8: (2, 0b11)}[src.index]
        self.code.append(instr << 12 | src.reg << 8 | dst.mode << 7 | b << 6 | src.mode << 4 | dst.reg)

        if not special_constant and (src.mode == M_INDEX or (src.mode == M_INCR and src.reg == R_PC)):
            self.code.append(src.index)
        if dst.mode == M_INDEX:
            self.code.append(dst.index)
        new_len = len(self.code)
        duration = {M_DIRECT: 1, M_INDEX: 3, M_INDIRECT: 2, M_INCR: 2}[src.mode] + (3 if dst.mode == M_INDEX else 0)
        if special_constant:
            duration = 4 if dst.mode == M_INDEX else 1
        return new_len - old_len, duration

    def one_operand_instr(self, instr, dst, b=False):
        old_len = len(self.code)
        self.code.append(0b000100 << 10 | instr << 7 | b << 6 | dst.mode << 4 | dst.reg)
        assert dst.mode in [M_DIRECT, M_INDEX, M_INDIRECT, M_INCR]
        if dst.mode == M_INDEX or (dst.mode == M_INCR and dst.reg == R_PC):
            self.code.append(dst.index)
        new_len = len(self.code)
        duration = {M_DIRECT: 1, M_INDEX: 4, M_INDIRECT: 4, M_INCR: 3}[dst.mode]
        if instr == ONE_INSTR['PUSH']:
            duration = {M_DIRECT: 2, M_INDEX: 4, M_INDIRECT: 4, M_INCR: 3}[dst.mode]
        elif instr == ONE_INSTR['CALL']:
            duration = {M_DIRECT: 3, M_INDEX: 5, M_INDIRECT: 5, M_INCR: 4}[dst.mode]
        return new_len - old_len, duration


    def jmp_instr(self, instr, lbl):
        offset = lbl.pos - len(self.code)
        mask = (1 << 10) - 1
        offset_masked = (offset & mask)
        self.code.append(0b001 << 13 | instr << 10 | offset_masked)
        duration = 1
        return 1, duration

    def label(self):
        lbl = label(len(self.code))
        return lbl


class label:
    def __init__(self, pos):
        self.pos = pos


def alu(instr, src, dst_in, flags_in, bw):
    def calc_alu(fn, *args):
        out, flags_upd = fn(*args)
        n_out = out[-1]  # negative equals sign bit
        z_out = And(*(~(a ^ b) for a, b in zip_all(Vector(out), Vector(0, len(out)))))
        c_out = ~z_out
        v_out = Signal(0)
        src[-1] & dst_in[-1]
        flags = {'c': c_out, 'n': n_out, 'z': z_out, 'v': v_out}
        flags.update(flags_upd)
        return out, flags


    def do_mov(src):
        return src, flags_in

    def do_add_sub(als, bls, c, instr):
        bls = Case(instr, {(INSTRS['ADD'], INSTRS['ADDC']): bls,
                           (INSTRS['SUB'], INSTRS['SUBC'], INSTRS['CMP']): Vector(Negate(bls))})
        sls, c_out = KoggeStoneAdder(als, bls, c)
        v_out = (als[-1] ^ sls[-1]) & (bls[-1] ^ sls[-1])
        return (Vector(sls)), {'c': c_out, 'v': v_out}

    def do_dadd(als, bls, c):
        sls, c_out = DecimalAdder(als, bls, c)
        out = Vector(sls)
        return out, {'c': c_out}

    def do_and(src, dst_in):
        out = src & dst_in
        return out, {}

    def do_xor(src, dst_in):
        out = src ^ dst_in
        return out, {'v': src[-1] & dst_in[-1]}

    def do_bic(src, dst_in):
        out = ~src & dst_in
        return out, flags_in

    def do_bis(src, dst_in):
        out = src | dst_in
        return out, flags_in

    def do_rrot(in_val, msb):
        out = Vector(in_val[1:] + [msb])
        return out, {'c': in_val[0],
                     'v': ~in_val[-1] & msb}

    def do_swpb(src):
        out = Vector(src[8:16] + src[0:8]) if len(src) == 16 else src
        return out, flags_in

    def do_sxt(src):
        out = Vector(src[:7] + ([src[7]] * (len(src) - 7)))
        return out, {}

    def do_jmp(src, dst_in):
        out = src + dst_in
        return out, flags_in

    def op(src, dst_in):
        c_in = flags_in['c']
        dst_out, flags_out = Case(instr, {  # Two arg instrs
                                            (INSTRS['MOV'], INSTRS['PUSH'], INSTRS['CALL']): calc_alu(do_mov, src),
                                            (INSTRS['ADD'], INSTRS['ADDC'], INSTRS['SUB'], INSTRS['SUBC'],
                                             INSTRS['CMP']):
                                                calc_alu(do_add_sub, src, dst_in,
                                                         Case(instr,
                                                              {(INSTRS['ADD'], INSTRS['SUB'], INSTRS['CMP']): Signal(),
                                                               (INSTRS['ADDC'], INSTRS['SUBC']): c_in}),
                                                         instr),
                                            INSTRS['BIC']: calc_alu(do_bic, src, dst_in),
                                            INSTRS['BIS']: calc_alu(do_bis, src, dst_in),
                                            (INSTRS['AND'], INSTRS['BIT']): calc_alu(do_and, src, dst_in),
                                            INSTRS['XOR']: calc_alu(do_xor, src, dst_in),
                                            INSTRS['DADD']: calc_alu(do_dadd, src, dst_in, c_in),
                                            # One arg instrs
                                            (INSTRS['RRA'], INSTRS['RRC']): calc_alu(do_rrot, src,
                                                                                     Case(instr,
                                                                                          {INSTRS['RRA']: src[-1],
                                                                                           INSTRS['RRC']: c_in})),
                                            INSTRS['SWPB']: calc_alu(do_swpb, src),
                                            INSTRS['SXT']: calc_alu(do_sxt, src),
                                            (INSTRS['JNZ'], INSTRS['JZ'], INSTRS['JNC'], INSTRS['JC'],
                                             INSTRS['JN'], INSTRS['JGE'], INSTRS['JL'], INSTRS['JMP']):
                                                calc_alu(do_jmp, src, dst_in)})
        return Case(instr, {(INSTRS['CMP'], INSTRS['BIT']): dst_in}, dst_out), flags_out

    dst_out, flags_out = Case(bw, {
        BYTE_WORD['BYTE']: (lambda dst_out, flags_out: (Vector(dst_out[:8] + dst_in[8:]), flags_out))(
            *op(Vector(src[:8]), Vector(dst_in[:8]))),
        BYTE_WORD['WORD']: op(src, dst_in)
    })

    return dst_out, flags_out


STATES = Enum('FETCH', 'SRC_INDIRECT', 'SRC_INCR', 'SRC_IDX', 'DST_IDX', 'DST_GET', 'DST_PUT', 'SRC_GET',
              'PUSH', 'CALL1', 'CALL2')


def CPU(mem_init, clk):
    reset = FeedbackSignal()

    dst_in = FeedbackVector(0, 16)
    sr_in = FeedbackVector(0, 16)

    pc_reg = FeedbackVector(0, 16)
    sp_reg = FeedbackVector(0, 16)
    sr_reg = FeedbackVector(0, 16)

    isWrite = Signal()

    prev_src = FeedbackVector(0, 16)
    prev_dst = FeedbackVector(0, 16)
    prev_idx_addr = FeedbackVector(0, 16)

    prev_instr_word = FeedbackVector(0, 16)
    prev_mem_in = FeedbackVector(0, 16)
    feedback_src_out = FeedbackVector(0, 16)

    state = FeedbackVector(STATES['FETCH'])

    mem_addr = Case(state, {STATES['FETCH']: pc_reg,
                            STATES['SRC_INDIRECT']: feedback_src_out,
                            STATES['SRC_INCR']: feedback_src_out,
                            STATES['SRC_IDX']: pc_reg,
                            STATES['SRC_GET']: prev_idx_addr,
                            STATES['DST_IDX']: pc_reg,
                            STATES['DST_GET']: prev_idx_addr,
                            STATES['DST_PUT']: prev_idx_addr,
                            STATES['PUSH']: sp_reg,
                            STATES['CALL1']: sp_reg,
                            STATES['CALL2']: prev_idx_addr})

    mem_out = Memory(mem_addr[1:8], prev_mem_in,
                     TrueInCase(state, (STATES['DST_PUT'], STATES['PUSH'], STATES['CALL1'])), mem_init)

    instr_word = Case(state, {STATES['FETCH']: mem_out}, default=prev_instr_word)
    prev_instr_word.connect(FlipFlops(instr_word, clk))

    instr, inst_type, src_reg, dst_reg, inst_bw, a_s, a_d, offset = decodeInstr(instr_word)

    flags_in = {v: sr_reg[FLAGS_SR_BITS[v]] for v in FLAGS_SR_BITS}

    dst_wr = (TrueInCase(state, {STATES['FETCH']:
                                     Case(inst_type, {I_TYPES['IT_TWO']:
                                                          TrueInCase(a_s, {
                                                          SRC_M['M_DIRECT']: TrueInCase(a_d, DST_M['M_DIRECT']),
                                                          (SRC_M['M_INDEX'], SRC_M['M_INDIRECT'], SRC_M['M_INCR']):
                                                              TrueInCase(src_reg, {(Vector(2, 4), Vector(3, 4)):
                                                                                       TrueInCase(a_d, DST_M[
                                                                                           'M_DIRECT'])})}),
                                                      I_TYPES['IT_ONE']: TrueInCase(a_s, SRC_M['M_DIRECT'])}),
                                 (STATES['SRC_INDIRECT'], STATES['SRC_INCR'], STATES['SRC_GET']):
                                     TrueInCase(inst_type, {I_TYPES['IT_TWO']: TrueInCase(a_d, DST_M['M_DIRECT'])}),
                                 STATES["DST_GET"]: TrueInCase(inst_type, I_TYPES['IT_TWO'])}) & \
              TrueInCase(inst_type, {I_TYPES['IT_TWO']: TrueInCase(a_d, DST_M['M_DIRECT']),
                                     I_TYPES['IT_ONE']: TrueInCase(a_s, SRC_M['M_DIRECT'])}) & \
              ~TrueInCase(instr, INSTRS['PUSH'])) | \
             TrueInCase(instr, {INSTRS['PUSH']: TrueInCase(state, STATES['PUSH']),
                                INSTRS['CALL']: TrueInCase(state, (STATES['CALL1'], STATES['CALL2'])),
                                INSTRS['JMP']: Signal(True),
                                INSTRS['JNZ']: ~flags_in['z'],
                                INSTRS['JZ']: flags_in['z'],
                                INSTRS['JNC']: ~flags_in['c'],
                                INSTRS['JC']: flags_in['c'],
                                INSTRS['JN']: flags_in['n'],
                                INSTRS['JGE']: ~(flags_in['n'] ^ flags_in['v']),
                                INSTRS['JL']: flags_in['n'] ^ flags_in['v'],
                                })

    src_out, dst_out, regs, src_incr = \
        RegisterFile(pc_incr=Case(state, {STATES['FETCH']: Signal(True),
                                          STATES['SRC_IDX']: Signal(True),
                                          STATES['DST_IDX']: Signal(True)}),
                     src_reg=src_reg,
                     dst_reg=Case(state, {(STATES['PUSH'], STATES['CALL1']): Vector(1, 4),
                                          STATES['CALL2']: Vector(0, 4)}, dst_reg),
                     src_incr=Case(state, {STATES['SRC_INCR']: Signal(True)}),
                     src_mode=a_s,
                     dst_in=Case(inst_bw,
                                 {BYTE_WORD['BYTE']: Vector(dst_in[:8] + [Signal()] * 8),
                                  BYTE_WORD['WORD']: dst_in}),
                     sr_in=sr_in,
                     dst_wr=dst_wr,
                     bw=inst_bw,
                     clk=clk,
                     reset=reset)
    feedback_src_out.connect(src_out)
    pc_reg.connect(regs[0])
    sp_reg.connect(regs[1] + Vector(-2))
    sr_reg.connect(regs[2])

    idx_addr = Case(state, {STATES['SRC_IDX']: src_out + mem_out,
                            STATES['DST_IDX']: dst_out + mem_out,
                            STATES['SRC_INCR']: src_incr,
                            STATES['SRC_INDIRECT']: src_out}, default=prev_idx_addr)
    prev_idx_addr.connect(FlipFlops(idx_addr, clk))

    src = Case(state, {STATES['FETCH']: Case(inst_type, {I_TYPES['IT_JUMP']:
                                                             Vector([Signal()] + offset[:] + [offset[-1]] * 5)},
                                             src_out),
                       (STATES['SRC_INDIRECT'], STATES['SRC_INCR'], STATES['SRC_GET']): mem_out}, default=prev_src)
    dst = Case(state, {STATES['FETCH']: dst_out,
                       STATES['DST_GET']:
                           Case(inst_bw,
                                {BYTE_WORD['BYTE']: If(mem_addr[0], Vector(mem_out[8:] + mem_out[:8]), mem_out),
                                 BYTE_WORD['WORD']: mem_out})}, default=prev_dst)

    prev_src.connect(FlipFlops(src, clk))
    prev_dst.connect(FlipFlops(dst, clk))

    alu_out, flags_out = alu(instr, src, dst, flags_in, inst_bw)

    REV_FLAGS_SR_BITS = {FLAGS_SR_BITS[k]: k for k in FLAGS}

    def flag_for_bit(n, b):
        if n in REV_FLAGS_SR_BITS:
            res = flags_out[REV_FLAGS_SR_BITS[n]]
            return res
        else:
            return b

    status_reg_out = Vector(flag_for_bit(n, b) for n, b in enumerate(regs[2]))

    mem_in = Case(instr, {INSTRS['CALL']: pc_reg},
                  Case(inst_bw, {BYTE_WORD['BYTE']: If(mem_addr[0], Vector(alu_out[8:] + alu_out[:8]), dst_in),
                                 BYTE_WORD['WORD']: alu_out}))

    prev_mem_in.connect(FlipFlops(mem_in, clk))

    dst_in.connect(Case(state, {(STATES['PUSH'], STATES['CALL1']): sp_reg}, alu_out))
    sr_in.connect(status_reg_out)

    next_dst_state = Case(a_d, {DST_M['M_DIRECT']: STATES['FETCH'],
                                DST_M['M_INDEX']: STATES['DST_IDX']})
    next_state = EnumVector(STATES,
                            Case(state, {STATES['FETCH']:
                                             Case(inst_type, {I_TYPES['IT_TWO']:
                                                                  Case(a_s, {SRC_M['M_DIRECT']: next_dst_state,
                                                                             SRC_M['M_INCR']:
                                                                                 Case(src_reg, {(Vector(2, 4), Vector(3,
                                                                                                                      4)): next_dst_state},
                                                                                      STATES['SRC_INCR']),
                                                                             SRC_M['M_INDIRECT']:
                                                                                 Case(src_reg, {(Vector(2, 4), Vector(3,
                                                                                                                      4)): next_dst_state},
                                                                                      STATES['SRC_INDIRECT']),
                                                                             SRC_M['M_INDEX']:
                                                                                 Case(src_reg,
                                                                                      {Vector(3, 4): next_dst_state},
                                                                                      STATES['SRC_IDX'])}),
                                                              I_TYPES['IT_ONE']:
                                                                  Case(a_s, {SRC_M['M_DIRECT']:
                                                                                 Case(instr,
                                                                                      {INSTRS['PUSH']: STATES['PUSH'],
                                                                                       INSTRS['CALL']: STATES['CALL1']},
                                                                                      STATES['FETCH']),
                                                                             SRC_M['M_INCR']: STATES['SRC_INCR'],
                                                                             SRC_M['M_INDIRECT']: STATES[
                                                                                 'SRC_INDIRECT'],
                                                                             SRC_M['M_INDEX']: STATES['SRC_IDX']}),
                                                              I_TYPES['IT_JUMP']: STATES['FETCH']}),
                                         (STATES['SRC_INCR'], STATES['SRC_INDIRECT'], STATES['SRC_GET']):
                                             Case(inst_type, {I_TYPES['IT_TWO']: next_dst_state,
                                                              I_TYPES['IT_ONE']: Case(instr,
                                                                                      {INSTRS['PUSH']: STATES['PUSH'],
                                                                                       INSTRS['CALL']: STATES['CALL1']},
                                                                                      STATES['DST_PUT'])}),
                                         STATES['SRC_IDX']: STATES['SRC_GET'],
                                         STATES['DST_IDX']: STATES['DST_GET'],
                                         STATES['DST_GET']: STATES['DST_PUT'],
                                         STATES['DST_PUT']: STATES['FETCH'],
                                         STATES['CALL1']: STATES['CALL2'],
                                         STATES['CALL2']: STATES['FETCH']}))
    state.connect(FlipFlops(next_state, clk))

    print calcAreaDelay(mem_addr[:] + mem_in[:] + [isWrite])
    return {'state': state, 'src_reg': src_reg, 'dst_reg': dst_reg,
            'src_out': src_out, 'dst_out': dst_out, 'mem_addr': Vector(mem_addr), 'mem_out': mem_out,
            'regs': regs, 'src': src, 'dst': dst, 'dst_in': dst_in, 'dst_wr': dst_wr, 'idx_addr': idx_addr,
            'instr': instr, 'sr_in': sr_in, 'mem_in': mem_in}
