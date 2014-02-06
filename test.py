'''
Created on Feb 6, 2014

@author: matrix1
'''
import unittest
from PyCircuit import Signal, FullAdder, SRLatch, DLatch, DFlipFlop, Not


class Test(unittest.TestCase):


    def test_FullAdder(self):
        a, b, c = Signal(), Signal(), Signal()
        signals = FullAdder(a, b, c)
        
        self.assertEquals([sig.value for sig in signals], [False, False])
        a.set()  # sum should be 1 (sum, carry) 
        self.assertEquals([sig.value for sig in signals], [True, False])
        b.set() # sum = 2
        self.assertEquals([sig.value for sig in signals], [False, True])
        c.set() # sum = 3
        self.assertEquals([sig.value for sig in signals], [True, True])




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
 
    

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()