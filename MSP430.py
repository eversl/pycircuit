'''
Created on Mar 21, 2014

@author: leonevers
'''
from PyCircuit import Vector, Signal, Memory, Decoder, DFlipFlop, Multiplexer,\
    FlipFlops, FeedbackVector, If, zip_all, FeedbackSignal, Or, calcAreaDelay,\
    Enum, Case, EnumVector, KoggeStoneAdder

def RegisterFile(pc_incr, src_reg, dst_reg, src_incr, dst_in, dst_wr, bw, clk, reset):
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
        
    src_out = Vector([Or(*l) for l in zip(*(src_line & q_out for src_line, q_out in zip_all(src_lines, q_outs)))])
    dst_out = Vector([Or(*l) for l in zip(*(dst_line & q_out for dst_line, q_out in zip_all(dst_lines, q_outs)))])
                      
    for out, old_out in zip_all(q_outs, old_outs):
        old_out.connect(FlipFlops(If(pc_incr, out + Vector(2, 16), out) if out is q_outs[0] else out, clk))
    
    incr_op = If(~ bw | src_lines[0] | src_lines[1], Vector(2,16), Vector(1,16))

    src_incr_val.connect(FlipFlops(src_out + incr_op, clk))
    src_incr_lines.connect(FlipFlops((src_incr[0] & src_line for src_line in src_lines), clk))
    dst_wr_lines.connect(FlipFlops((dst_wr & dst_line for dst_line in dst_lines), clk))
    dst_wr_val.connect(FlipFlops(dst_in, clk))
    return src_out, dst_out, q_outs    

I_TYPES = Enum('IT_ONE', 'IT_JUMP', 'IT_TWO')
IT_TWO_INS = Enum('_0', '_1', '_2', '_3', )
BYTE_WORD = Enum('WORD', 'BYTE')
def decodeInstr(word):
    inst_type = If(word[15] | (~word[15] & word[14]), I_TYPES['IT_TWO'], If(word[13], I_TYPES['IT_JUMP'],               I_TYPES['IT_TWO']))
    instr = EnumVector(INSTRS, If(word[15] | (~word[15] & word[14]), Vector(word[12:]), If(word[13], Vector(word[10:13] + [Signal()]), Vector(word[7:10] + [Signal()]))))
    src_reg = Vector(word[8:12])
    dst_reg = Vector(word[0:4])
    bw = EnumVector(BYTE_WORD, word[6])
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
    # register mode
    def __init__(self, reg):
        self.reg = reg
        self.mode = M_DIRECT  
        
    # indirect autoincrement mode
    def __pos__(self):
        assert (self.mode == M_DIRECT)
        self.mode = M_INCR 
        return self

    # Indirect register mode
    def __invert__(self):
        assert (self.mode == M_DIRECT)
        self.mode = M_INDIRECT 
        return self

    # indexed mode
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
    # Immediate mode
    def __init__(self, num):
        self.index = num
        self.n_mode = M_IMM 
        self.mode = M_INCR
        self.reg = R_PC 

    # Symbolic mode
    def __pos__(self):
        assert (self.n_mode == M_IMM)
        self.n_mode = M_SYM 
        self.mode = M_INDEX
        self.reg = R_PC 
        return self
    
    # Absolute mode
    def __inv__(self):
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
INSTRS = Enum(TWO_INSTR)

B = True
class CodeSequence(object):
    def __init__(self):
        self.code = []
    
    def two_operand_instr(self, instr, src, dst, b=False):
            
        self.code.append(instr << 12 | src.reg << 8 | dst.mode << 7 | b << 6 | src.mode << 4 | dst.reg)
        if src.mode == M_INDEX or (src.mode == M_INCR and src.reg == R_PC):
            self.code.append(src.index)
        if dst.mode == M_INDEX:
            self.code.append(dst.index)
        
    def MOV(self, src, dst, b=False):
        self.two_operand_instr(TWO_INSTR['MOV'], src, dst, b)
    def ADD(self, src, dst, b=False):
        self.two_operand_instr(TWO_INSTR['ADD'], src, dst, b)
    def ADDC(self, src, dst, b=False):
        self.two_operand_instr(TWO_INSTR['ADDC'], src, dst, b)
    def SUBC(self, src, dst, b=False):
        self.two_operand_instr(TWO_INSTR['SUBC'], src, dst, b)
    def SUB(self, src, dst, b=False):
        self.two_operand_instr(TWO_INSTR['SUB'], src, dst, b)
    def CMP(self, src, dst, b=False):
        self.two_operand_instr(TWO_INSTR['CMP'], src, dst, b)
    def DADD(self, src, dst, b=False):
        self.two_operand_instr(TWO_INSTR['DADD'], src, dst, b)
    def BIT(self, src, dst, b=False):
        self.two_operand_instr(TWO_INSTR['BIT'], src, dst, b)
    def BIC(self, src, dst, b=False):
        self.two_operand_instr(TWO_INSTR['BIC'], src, dst, b)
    def BIS(self, src, dst, b=False):
        self.two_operand_instr(TWO_INSTR['BIS'], src, dst, b)
    def XOR(self, src, dst, b=False):
        self.two_operand_instr(TWO_INSTR['XOR'], src, dst, b)
    def AND(self, src, dst, b=False):
        self.two_operand_instr(TWO_INSTR['AND'], src, dst, b)
    

def alu(instr, src, dst_in, flags_in, bw):
    def op(src, dst_in):
        c_in = flags_in['c']
        dst_out, c_out = Case(instr, {INSTRS['MOV']: (src, c_in),
                                      INSTRS['ADD']: KoggeStoneAdder(src, dst_in, c_in),
                                      INSTRS['ADDC']: KoggeStoneAdder(src, dst_in, c_in)})
        v_out = (src[-1] & dst_in[-1] & ~ dst_out) | ~src[-1] & ~dst_in[-1] & dst_out   # for add operations
        return src + dst_in, flags_in
    
    dst_out, flags_out = {BYTE_WORD['BYTE']: (lambda dst_out, flags_out: (dst_out[:8] + dst_in[8:], flags_out)) (*op(src[:8], dst_in[:8])), 
                          BYTE_WORD['WORD']: op(src, dst_in)}
     
    return dst_out, flags_out


def CPU(mem_init, clk):    
    bw = FeedbackVector(BYTE_WORD['WORD'])
    reset = FeedbackSignal()
    
    dst_in = FeedbackVector(0, 16)
    dst_wr = FeedbackVector(0, 1)
    
    isWrite = Signal()
        
    current_src = FeedbackVector(0, 16)
    current_dst = FeedbackVector(0, 16)
    current_idx_addr = FeedbackVector(0, 16)

    STATES = Enum('FETCH', 'SRC_INDIRECT', 'SRC_INCR', 'SRC_IDX', 'DST_IDX', 'DST_GET', 'DST_PUT', 'SRC_GET')
    
    state = FeedbackVector(STATES['FETCH'])
    
    src_reg = FeedbackVector(0, 4)
    dst_reg = FeedbackVector(0, 4)
    src_out, dst_out, regs = RegisterFile(Case(state, {STATES['FETCH']: Signal(True),
                                                       STATES['SRC_IDX']: Signal(True),
                                                       STATES['DST_IDX']: Signal(True)}), 
                                          src_reg, dst_reg, 
                                          Case(state, {STATES['SRC_INCR']: Signal(True)}), 
                                          Case(bw, {BYTE_WORD['BYTE']: Vector(dst_in[:8] + [Signal()] * 8),
                                                    BYTE_WORD['WORD']: dst_in}), 
                                          dst_wr[0], bw[0], clk, reset)    

    mem_addr = Case(state, {STATES['FETCH']: regs[0], 
                            STATES['SRC_INDIRECT']: src_out,
                            STATES['SRC_INCR']: src_out,
                            STATES['SRC_IDX']: regs[0],
                            STATES['SRC_GET']: current_idx_addr,
                            STATES['DST_IDX']: regs[0],
                            STATES['DST_GET']: current_idx_addr,
                            STATES['DST_PUT']: current_idx_addr}) 
    
    mem_in = Case(bw, {BYTE_WORD['BYTE']: If(mem_addr[0], Vector(dst_in[8:] + dst_in[:8]), dst_in),
                       BYTE_WORD['WORD']: dst_in})
    mem_out = Memory(mem_addr[1:6], mem_in, 
                     Case(state, {STATES['DST_PUT']: Signal(True)}), mem_init)
    
    idx_addr = Case(state, {STATES['SRC_IDX']: src_out + mem_out,
                            STATES['DST_IDX']: dst_out + mem_out}, default=current_idx_addr)

    current_instr = FeedbackVector(0, 16)
    
    instr_state = Case(state, {STATES['FETCH']:mem_out}, default=current_instr)
    current_instr.connect(FlipFlops(instr_state, clk))
    
    instr, inst_type, instr_src, instr_dst, inst_bw, a_s, a_d, offset = decodeInstr(instr_state)

    src_reg.connect(instr_src)
    dst_reg.connect(instr_dst)

    src = Case(state, {STATES['FETCH']: src_out, 
                       (STATES['SRC_INDIRECT'], STATES['SRC_INCR'], STATES['SRC_GET']): mem_out}, default=current_src)
    dst = Case(state, {STATES['FETCH']: dst_out,
                       STATES['DST_GET']: 
                       Case(bw, {BYTE_WORD['BYTE']: If(mem_addr[0], Vector(mem_out[8:] + mem_out[:8]), mem_out),
                                 BYTE_WORD['WORD']: mem_out})}, default=current_dst)

    current_src.connect(FlipFlops(src, clk))
    current_dst.connect(FlipFlops(dst, clk))
    current_idx_addr.connect(FlipFlops(idx_addr, clk))

    do_alu = Case(state, {STATES['FETCH']: 
                          Case(inst_type, {I_TYPES['IT_TWO']: 
                                           Case(a_s, {SRC_M['M_DIRECT']: 
                                                      Case(a_d, {DST_M['M_DIRECT']: Signal(True)})})}),
                          (STATES['SRC_INDIRECT'], STATES['SRC_INCR'], STATES['SRC_GET']):
                          Case(inst_type, {I_TYPES['IT_TWO']:
                                           Case(a_d, {DST_M['M_DIRECT']: Signal(True)})})
                          })
    
    flags_in = {'c': regs[2][0]}
    alu_res = alu(instr, src, dst, flags_in, bw)
    
    dst_in.connect(alu_res)
    dst_wr.connect(do_alu & Case(a_d, {DST_M['M_DIRECT']: Signal(True)}))
    
    bw.connect(Case(state, {STATES['FETCH']: inst_bw}, default=bw))

    next_dst_state = Case(inst_type, {I_TYPES['IT_TWO']: 
                                      Case(a_d, {DST_M['M_DIRECT']:STATES['FETCH'],            
                                                 DST_M['M_INDEX']:STATES['DST_IDX']})})
    next_state = Case(state, {STATES['FETCH']: 
                              Case(inst_type, {I_TYPES['IT_TWO']: 
                                               Case(a_s, {SRC_M['M_DIRECT']: next_dst_state, 
                                                          SRC_M['M_INCR']: STATES['SRC_INCR'],
                                                          SRC_M['M_INDIRECT']: STATES['SRC_INDIRECT'],
                                                          SRC_M['M_INDEX']: STATES['SRC_IDX']})}),
                              STATES['SRC_INCR']: next_dst_state,
                              STATES['SRC_INDIRECT']: next_dst_state,
                              STATES['SRC_IDX']: STATES['SRC_GET'],
                              STATES['SRC_GET']: next_dst_state,
                              STATES['DST_IDX']: STATES['DST_GET'],
                              STATES['DST_GET']: STATES['DST_PUT'],
                              STATES['DST_PUT']: STATES['FETCH']})
    state.connect(FlipFlops(next_state, clk))
    print calcAreaDelay(mem_addr[:] + mem_in[:] + [isWrite])
    return {'state': state, 'src_reg': src_reg, 'dst_reg': dst_reg, 'bw': bw, 
            'src_out': src_out, 'dst_out': dst_out, 'mem_addr': Vector(mem_addr), 'mem_out': mem_out,
            'pc': regs[0], 'src': src, 'dst': dst, 'dst_in': dst_in, 'dst_wr': dst_wr, 'idx_addr': idx_addr,
            'instr': instr}

    
    