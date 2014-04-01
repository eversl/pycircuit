'''
Created on Mar 21, 2014

@author: leonevers
'''
from PyCircuit import Vector, Signal, Memory, Decoder, DFlipFlop, Multiplexer,\
    FlipFlops, FeedbackVector, If, zip_all, FeedbackSignal, Or, calcAreaDelay

def RegisterFile(src_reg, dst_reg, dst_in, src_incr, dst_wr, bw, clk, reset):
    dst_wordlines = Decoder(dst_reg)
    src_wordlines = Decoder(src_reg)
    
    src_incr_val = FeedbackVector(0, 16)
    old_outs = [FeedbackVector(0, 16) for _ in dst_wordlines]
    
    q_outs = [Vector(If(reset, Vector(0, 16), 
                        If(src_incr & src_wordline, src_incr_val,
                           If(dst_wr & dst_wordline, dst_in,
                              old_out)))) 
              for src_wordline, dst_wordline, old_out in zip_all(src_wordlines, dst_wordlines, old_outs)]
    
    old_outs = [FlipFlops(out, clk) for out in q_outs]
    
    
    src_out = Vector([Or(*l) for l in zip(*(src_wordline & q_out for src_wordline, q_out in zip_all(src_wordlines, q_outs)))])
    dst_out = Vector([Or(*l) for l in zip(*(dst_wordline & q_out for dst_wordline, q_out in zip_all(dst_wordlines, q_outs)))])
                      
    incr_op = Vector(2,16) if ~ bw or src_wordlines[0] or src_wordlines[1] else Vector(1,16)
    src_incr_val.connect(FlipFlops(src_out + incr_op, clk))
    
    return src_out, dst_out    


def decodeInstr(word):
    instr = Vector(word[12:])
    src_reg = Vector(word[8:12])
    dst_reg = Vector(word[0:4])
    
    return instr, src_reg, dst_reg


def CPU(clk):
    
    src_reg = FeedbackVector(0, 4)
    print src_reg
    dst_reg = FeedbackVector(0, 4)
    src_a = Vector(3, 2)
    dst_a = Vector(0, 1)
    bw = FeedbackSignal()
    reset = FeedbackSignal()
    
    dst_in = FeedbackVector(0, 16)
    src_incr = FeedbackSignal()
    dst_wr = FeedbackSignal()
    
    src_out, dst_out = RegisterFile(src_reg, dst_reg, dst_in, src_incr, dst_wr, bw, clk, reset)
        
    mem_addr = src_out[1:4]
    mem_in = Vector(0, 16)
    isWrite = Signal()
        
    mem_out = Memory(mem_addr, mem_in, isWrite)
    print calcAreaDelay(mem_addr[:] + mem_in[:] + [isWrite])
    
    instr, instr_src, instr_dst = decodeInstr(mem_out)
    src_reg.connect(instr_src)
    dst_reg.connect(instr_dst)
    
    return {'src_reg': src_reg, 'dst_reg': dst_reg, 'src_out': src_out, 'dst_out': dst_out, 'mem_addr': mem_addr, 'mem_out': mem_out}

    
    
    
    