'''
Created on Feb 4, 2014

@author: leone
'''


def zip_all(*args):
    for i in xrange(0, len(args)):
        for j in xrange(i + 1, len(args)):
            if len(args[i]) != len(args[j]):
                raise TypeError('Vectors are not of equal size')
    return zip(*args)
       
    
class Signal(object):
    __slots__ = ['fanout', 'value']
    def __init__(self, initval=False):
        self.value = bool(initval)
        self.fanout = []
        
    def __xor__(self, other):
        return Xor(self, other)
    
    def __or__(self, other):
        return Or(self, other)
    
    def __and__(self, other):
        return And(self, other)
    
    def __invert__(self):
        return Not(self)
    
    def __mul__(self, other):
        return Vector([self] * other)

    def __repr__(self):
        return '{0}({1})'.format(type(self).__name__,self.value)
        
    def set(self, value=True):
        if value != self.value:
            self.value = value
            for c in self.fanout:
                c()
    
    def reset(self):
        self.set(False)
        
        
class FeedbackSignal(Signal):
    def connect(self, sig):
        self.sig = sig
        sig.fanout.append(self)
        self()
    
    def __call__(self):
        self.set(self.sig.value)

class Vector():
    def __init__(self, value, bits=16):
        try:
            value = (s for s in value)
        except TypeError:
            self.ls = intToSignals(int(value), bits)
        else:
            self.ls = [s if isinstance(s, Signal) else Signal(s) for s in value]
        
    def __repr__(self):
        return 'Vector({0}, {1})'.format(int(self), len(self))

    def __len__(self):
        return len(self.ls)
        
    def __getitem__(self, key):
        return self.ls[key]
        
    def __iter__(self):
        return iter(self.ls)
        
    def __add__(self, other):
        return Vector(RippleCarryAdder(self, other)[0])
        
    def __sub__(self, other):
        return Vector(RippleCarryAdder(self, -other)[0])
        
    def __mul__(self, other):
        return Vector(Multiplier(self, other))
        
    def __floordiv__(self, other):
        return NotImplemented
        
    def __mod__(self, other):
        return NotImplemented
        
    def __divmod__(self, other):
        return NotImplemented
        
    def __lshift__(self, other):
        return NotImplemented
        
    def __rshift__(self, other):
        return NotImplemented
        
    def __and__(self, other):
        return Vector(s & o for s, o in zip_all(self, other))
        
    def __xor__(self, other):
        return Vector(s ^ o for s, o in zip_all(self, other))
            
    def __or__(self, other):
        return Vector(s | o for s, o in zip_all(self, other))
        
    def __neg__(self):
        inv = ~self
        return inv + Vector(1, len(self))
        
    def __pos__(self):
        return self
        
    def __abs__(self):
        return Vector(If(self[-1], -self, +self))
        
    def __invert__(self):
        return Vector(~a for a in self)
    
    def __int__(self):
        return signalsToInt(self.ls)
        
    def __setitem__(self, key, value):
        if isinstance(key, slice):
            signals = self.ls[key]
        if isinstance(value, int):
            bits = intToBits(value, len(signals))
        else:
            bits = value    
        for b, s in zip(bits, signals): 
            s.set(b)


def intToBits(num, bits):
    return (((num >> i) & 1) == 1 for i in xrange(bits))

def intToSignals(num, bits):
    return [Signal(b) for b in intToBits(num, bits)]

def signalsToInt(ls, signed=True):
    num = 0
    for i in xrange(len(ls)):
        num |= (1 << i if ls[i].value else 0)  
    if signed and ls[-1].value:
        num -= (1 << len(ls))
    return num
    



class CircuitOper(Signal):
    def __init__(self, *args):
        Signal.__init__(self)
        self.args = args
        
        for arg in args:
            arg.fanout.append(self)
        self.__call__()
        
    def __call__(self):
        self.set(self.func(*(arg.value for arg in self.args)))
        
    def func(self):
        raise NotImplementedError()

class And(CircuitOper):
    def func(self, a, b):
        return a & b 

class Or(CircuitOper):
    def func(self, *a):
        return reduce(lambda p, q: p | q, a)

class Xor(CircuitOper):
    def func(self, a, b):
        return a ^ b

class Not(CircuitOper):
    def func(self, a):
        return not a

class Nor(CircuitOper):
    def func(self, a, b):
        return not (a | b)


def HalfAdder(a, b):
    s = a ^ b
    c = a & b
    return s, c

def FullAdder(a, b, c_in):
    s1, c1 = HalfAdder(a, b)
    s2, c2 = HalfAdder(s1, c_in)
    c_out = c1 | c2
    return s2, c_out
     

def RippleCarryAdder(als, bls, c=Signal()):
    sls = []
    for a, b in zip_all(als, bls): 
        s, c = FullAdder(a, b, c)
        sls.append(s)
    return sls, c     


def KoggeStoneAdder(als, bls, c=Signal()):
    sls = []
    prop_gen = {}
    prop_gen[0] = [(a ^ b,  # propagate
                 a & b)   # generate
                for a, b in zip_all(als, bls)] 

    step = 1
    while step < len(prop_gen[0]):
        prop_gen[step] = []
        for i in xrange(len(prop_gen[0])):
            p_i, g_i = prop_gen[step/2][i]
            p_prev, g_prev = prop_gen[step/2][i-step] if i-step >= 0 else (Signal(0), Signal(0)) 
            prop_gen[step].append((p_i & p_prev, (p_i & g_prev) | g_i)) 
        step *= 2
        
    cls = [c] + [g for p,g in prop_gen[step/2]]
    for a, b, c in zip_all(als, bls, cls[:-1]): 
        s = a ^ b ^ c
        sls.append(s)
    return sls, cls[-1]

def Negate(als):
    sls, _ = RippleCarryAdder([Not(a) for a in als], [Signal()] * len(als), Signal(True))
    return sls

def Multiplier(als, bls, signed=True):
    if signed:
        als_sign = als[-1]
        bls_sign = bls[-1]
        als = If(als_sign, Negate(als), als)
        bls = If(bls_sign, Negate(bls), bls)
    else: 
        als = [a for a in als]  # to make sure Vectors are made to lists
        bls = [b for b in bls]
    sls, c = [Signal()] * (len(bls) - 1), Signal()
    for a in als:
        sls, c = RippleCarryAdder([a & b for b in bls], sls + [c])
        bls = [Signal()] + bls
    
    res = sls + [c]
    if signed:   
        sls_sign = als_sign ^ bls_sign
        res = If(sls_sign, Negate(res), res) 
    return res
     
     
def If(pred, cons, alt):
    return [(pred & c) | (Not(pred) & a) for c, a in zip_all(cons, alt)]
         
     
def SRLatch(s, r):
    q_fb = FeedbackSignal(not r.value)
    nq_fb = FeedbackSignal(not s.value)
    nq = Nor(s, q_fb)
    q = Nor(r, nq_fb)
    q_fb.connect(q)
    nq_fb.connect(nq)
    return q, nq
    
def DLatch(d, e):
    return SRLatch(s=d & e, r=Not(d) & e)    



def DFlipFlop(d, clk):
    master = DLatch(d=d, e=clk)
    slave = DLatch(d=master[0], e=Not(clk))
    return slave


def Decoder(a):
    if len(a) == 1:
        return [~a[-1], a[-1]]
    else:
        sub = Decoder(a[:-1])
        not_a = ~a[-1]
        return [not_a & d for d in sub] + [a[-1] & d for d in sub]
    
    
def Memory(mem_addr, data, isWrite):
    wordlines = Decoder(mem_addr)
    sr = [(isWrite & d, isWrite & ~d) for d in data]

    q_outs = [[SRLatch(mem_wr & s, mem_wr & r)[0] & mem_wr for s, r in sr] for mem_wr in wordlines]
    return Vector([Or(*l) for l in zip(*q_outs)])
        
    

def Multiplexer(sel, alts):
    enables = Decoder(sel)
    return Vector([Or(*[l & mem_wr for (l, mem_wr) in zip_all(ll, enables)]) for ll in zip_all(*alts)])


def RegisterFile(addr1, addr2, addr_w, data_w, clk_w):
    wordlines_w = Decoder(addr_w)
    
    q_outs = [[DFlipFlop(d, mem_wr & clk_w)[0] for d in data_w] for mem_wr in wordlines_w]
    
    data1 = Multiplexer(addr1, q_outs)                
    data2 = Multiplexer(addr2, q_outs)      
    return data1, data2    


def calcAreaDelay(inputs):
    levels = [set(inputs)]
    i = 0
    while len(levels[i]) > 0:
        levels.append(set(c 
        for inp in levels[i]
            for c in inp.fanout
                if not isinstance(c, FeedbackSignal)))

        i += 1
    return len(set().union(*levels)), i-1

