'''
Created on Feb 4, 2014

@author: leone
'''


def zip_all(*args):
    for i in xrange(0, len(args)):
        for j in xrange(i+1, len(args)):
            if len(args[i]) != len(args[j]):
                raise TypeError('Vectors are not of equal size')
    return zip(*args)
       
    
class Signal():
    def __init__(self, initval = False):
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
        return 'Signal({0})'.format(self.value)
        
    def set(self, value = True):
        if value != self.value:
            self.value = value
            for c in self.fanout:
                c()
    
    def reset(self):
        self.set(False)
        
        
class FeedbackSignal(Signal):
    def connect(self, sig):
        def inner():
            self.set(sig.value)
        sig.fanout.append(inner)
        inner()
    

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
        print key, value
        return self

def intToSignals(num, bits):
    return [Signal(((num >> i) & 1) == 1) for i in xrange(bits)]

def signalsToInt(ls, signed=True):
    num = 0
    for i in xrange(len(ls)):
        num |= (1 << i if ls[i].value else 0)  
    if signed and ls[-1].value:
        num -= (1 << len(ls))
    return num
    



def CircuitOper(func, *args):
    sig = Signal()
    def inner():
        sig.set(func(*[arg.value for arg in args]))
    for arg in args:
        arg.fanout.append(inner)
    inner()
    return sig

def And(a, b):
    return CircuitOper(lambda x, y: x & y, a, b)

def Or(a, b):
    return CircuitOper(lambda x, y: x | y, a, b)

def Xor(a, b):
    return CircuitOper(lambda x, y: x ^ y, a, b)

def Not(a):
    return CircuitOper(lambda x: not x , a)

def Nor(a, b):
    return CircuitOper(lambda x, y: not (x | y), a, b)


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

def Negate(als):
    sls, _ = RippleCarryAdder([Not(a) for a in als], [Signal()] * len(als), Signal(True))
    return sls

def Multiplier(als, bls, signed = True):
    if signed:
        als_sign = als[-1]
        bls_sign = bls[-1]
        als = If(als_sign, Negate(als), als)
        bls = If(bls_sign, Negate(bls), bls)
    else: 
        als = [a for a in als]  # to make sure Vectors are made to lists
        bls = [b for b in bls]
    sls, c = [Signal()] * (len(bls)-1), Signal()
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
    return SRLatch(s = d & e, r = Not(d) & e)    



def DFlipFlop(d, clk, s=Signal(0), r=Signal(0)):
    master = DLatch(d=d, e=clk)
    slave = DLatch(d=master[0], e=Not(clk))
    return slave

