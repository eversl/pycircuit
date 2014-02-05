'''
Created on Feb 4, 2014

@author: leone
'''
import inspect

class Signal():
    def __init__(self, initval = False):
        self.value = initval
        self.fanout = []
        
    def __xor__(self, other):
        return Xor(self, other)
    
    def __or__(self, other):
        return Or(self, other)
    
    def __and__(self, other):
        return And(self, other)
    
    def __str__(self):
        return 'Signal({0})'.format(self.value)
        
    def __repr__(self):
        return self.__str__()
        
    def set(self, value = True):
        if value != self.value:
            self.value = value
            for c in self.fanout:
                c()
        

class Circuit():
    def __init__(self, name, inputs):
        self.name = name
        self.inputs = inputs
        print 'Circuit', name, inputs
#         for i in inputs:
#             inputs[i].fanout.append(self)
            
    def __call__(self, *args):
        print 'call', args
            
    def update(self):
        print 'update', self.name
        

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

def HalfAdder(a, b):
    s = a ^ b
    c = a & b
    return s, c

def FullAdder(a, b, c_in):
    s1, c1 = HalfAdder(a, b)
    s2, c2 = HalfAdder(s1, c_in)
    c_out = c1 | c2
    return s2, c_out
     

if __name__ == '__main__':
    
    a, b, c = Signal(), Signal(), Signal()
    signals = FullAdder(a, b, c)
    
    a.set()
    print signals
    b.set()
    print signals
    c.set()
    print signals
    


