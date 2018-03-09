import unittest

from PyCircuit import ConstantVector


class Test(unittest.TestCase):
    def test_CPP(self):
        a = ConstantVector(0, 8) & ConstantVector(1, 8)
        print(a)
