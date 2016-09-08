'''
Created on Feb 4, 2014

@author: leone
'''
import inspect
from abc import abstractmethod, ABCMeta
from collections import Counter
from operator import __and__, __xor__, __or__, __invert__

sim_steps = 0


def zip_all(*args):
    for i in xrange(0, len(args)):
        for j in xrange(i + 1, len(args)):
            if len(args[i]) != len(args[j]):
                raise TypeError('Vectors are not of equal size: %i and %i' % (len(args[i]), len(args[j])))
    return zip(*args)


monitored = []
monitoredVectors = []


def monitor(val):
    global monitored
    if isinstance(val, Vector):
        monitored.extend(val.ls)
        monitoredVectors.append(val)
    else:
        monitored.extend(val)
        monitoredVectors.append(val)


def printMonitored():
    print "{:>8d}:  {}".format(sim_steps,
                               "".join(("{:^16}".format([k for k in v.enum if v.enum[k] == v])
                                        if v.enum else "{:^8}".format(signalsToInt(v.ls)))
                                       if isinstance(v, Vector) else
                                       "".join("-" if s.value else "_" for s in v)
                                       for v in monitoredVectors))



def simulate(signals, recur=None):
    global sim_steps
    if recur is None:
        recur = Counter()
    sim_steps += 1
    next_signals = set()
    for sig in signals:
        if isinstance(sig, FeedbackSignal):
            recur.update(sig)
            if recur[sig] >= 8:
                def findPrev(sigs, targ, oldSigs):
                    # print len(sigs)
                    if targ in sigs:
                        return [targ]
                    allArgs = [s.args for s in sigs]
                    allArgSet = set()
                    for s in allArgs:
                        allArgSet.update(s)
                    allArgSet.difference_update(oldSigs)
                    if not allArgSet:
                        return sigs
                    oldSigs.update(allArgSet)
                    path = findPrev(list(allArgSet), targ, oldSigs)
                    return path + [sigs[[n for n, a in enumerate(allArgs) if path[-1] in a][0]]]

                path = findPrev(sig.args, sig, set())
                # print "findPrev", len(path), sig.stack.f_back.f_code.co_name
                nexts = set.union(*(sig.eval() for sig in path))
                # recur[sig] = 0

                new_nexts = nexts
                next_signals.update(new_nexts)

                continue
        next_signals.update(sig.eval())
    global monitored
    if any(s in signals for s in monitored):
        printMonitored()
    if not next_signals:
        return
    simulate(next_signals, recur)


def oldSimplify(*args):
    """
    Replaces circuits by simpler, cheaper variants wherever it can by performing
    constant propagation, and replacing chained binary circuits by n-ary ones
    Returns a tuple with replacements of the input arguments
    :param args: all the signals to simplify
    """
    D = {True: STrue, False: SFalse, None: Signal(DontCare())}
    new_args = []
    for arg in args:
        if isinstance(arg, Vector):
            arg = Vector(a.oldSimplify(D) for a in arg)
        elif isinstance(arg, Signal):
            arg = arg.oldSimplify(D)
        else:
            arg = [a.oldSimplify(D) for a in arg]
        new_args.append(arg)

    DT = D[True]; del D[True]
    DF = D[False]; del D[False]
    DN = D[None]; del D[None]
    vecs = set(k.vec for k in D)
    print "simplify", len(vecs), len(D), len(set(D.values()))
    return tuple(new_args)

def simplify(*args, **kwargs):
    """
    Replaces circuits by simpler, cheaper variants wherever it can by performing
    constant propagation, and replacing chained binary circuits by n-ary ones
    Returns a tuple with replacements of the input arguments
    :param args: all the signals to simplify
    """
    if 'D' in kwargs:
        D = kwargs['D']
    else:
        D = {True: STrue, False: SFalse, None: DontCareSig}
    new_args = []
    for arg in args:
        if isinstance(arg, Vector):
            for a in arg.ls: a.simplify(D)
        elif isinstance(arg, Signal):
            arg = arg.simplify(D)
        else:
            arg = list(simplify(*arg, D=D))
        new_args.append(arg)

    if 'D' not in kwargs:
        DT = D[True]; del D[True]
        DF = D[False]; del D[False]
        DN = D[None]; del D[None]
    vecs = set([k.vec for k in D if hasattr(k, 'vec')])
    for v in vecs:
        if v is None: continue
        val = v.current()
        v.ls = [D[e] if e in D else e for e in v.ls]
        v.checkEqual(val)
    return tuple(new_args)


current_cache = {}


class Signal(object):
    def __init__(self):
        self.fanout = set()
        self.vec = None

    def __xor__(self, other):
        return Xor(self, other)

    def __or__(self, other):
        return Or(self, other)

    def __and__(self, other):
        return And(self, other)

    def __invert__(self):
        return Not(self)

    def __repr__(self):
        return '{0}({1}) [#{2:X}]'.format(type(self).__name__, self.value, id(self))

    def __len__(self):
        return 1

    def __nonzero__(self):
        return self.value

    def __iter__(self):
        return iter([self])

    def concat(self, *others):
        return Vector(self).concat(*others)

    def simulate(self):
        simulate([self])

    def eval(self):
        try:
            value = self.func(*(arg.value for arg in self.args))
        except TypeError:
            valueTrue = self.func(*(True if type(arg.value) is DontCare else arg.value for arg in self.args))
            valueFalse = self.func(*(False if type(arg.value) is DontCare else arg.value for arg in self.args))
            value = valueTrue if valueTrue == valueFalse else DontCare()
        if self.value == value:
            return []
        else:
            self.value = value
            return self.fanout


    def checkEqual(self, val):
        if val != self.value:
            print self, "current result", val, "not equal to", self.value

    def _setArgs(self, args, check=False):
        if check:
            assert not self.args
            value = self.value

        if args:
            for a in args:
                assert not isinstance(a, Vector)
            args = tuple(a if isinstance(a, Signal) else ConstantSignal(a) for a in args)
            for arg in args:
                arg.fanout.add(self)
            self.args = args
            self.value = None
            return self.eval()

        if check:
            assert self.value == value

    def _removeArgs(self, check=False):
        args = self.args
        if check:
            for arg in args:
                if not self in arg.fanout:
                    print "whoops!", self, arg, arg.fanout
                assert self in arg.fanout
            for f in self.fanout:
                assert self in f.args

        for arg in args:
            arg.fanout.discard(self)
        self.args = ()
        return args

    def simplifyTop(self, D):
        args = self._removeArgs(True)
        args = tuple(a.simplify(D) for a in args)
        return args

    def simplifyBottom(self, res, args, D):
        res._setArgs(args, True)
        if res != self:
            fanout = self.fanout
            for f in fanout:
                assert self in f.args
                f.args = tuple(res if a == self else a for a in f.args)
            res.fanout.update(fanout)
            self.fanout.clear()
            for arg in res.args:
                assert res in arg.fanout
        D[self] = res
        return res

    def func(self, *args):
        raise NotImplementedError

    def current(self):
        raise NotImplementedError


class ConstantSignal(Signal):
    class FanoutSet(set):
        def add(self, item):
            pass

        def update(self, *args, **kwargs):
            pass

        def __contains__(self, item):
            return True

    def __init__(self, value=False):
        Signal.__init__(self)
        self.args = ()
        self.fanout = ConstantSignal.FanoutSet()
        self.value = value.value if isinstance(value, Signal) else bool(value)

    def simplify(self, D):
        args = self.simplifyTop(D)
        res = D.get(self.value, D[None])
        return self.simplifyBottom(res, args, D)

    def func(self):
        return self.value

    def current(self):
        return self.value
    
STrue = ConstantSignal(True)
SFalse = ConstantSignal(False)

class TestSignal(Signal):
    def __init__(self, initval=False):
        Signal.__init__(self)
        self.args = ()
        self.value = initval

    def set(self, value=True):
        self.value = value.value if isinstance(value, Signal) else value
        self.simulate()

    def reset(self):
        self.set(False)

    def eval(self):
        return self.fanout

    def func(self):
        return self.value

    def current(self):
        return self.value

    def simplify(self, D):
        args = self.simplifyTop(D)
        return self.simplifyBottom(self, args, D)


class FeedbackSignal(Signal):
    def __init__(self, arg):
        Signal.__init__(self)
        self._setArgs((arg,))

    def simplify(self, D):
        if self in D:
            # print "found before", self
            return self
        # print "first time", self
        D[self] = self
        args = self.simplifyTop(D)
        return self.simplifyBottom(self, args, D)

    def connect(self, sig):
        self.stack = inspect.currentframe().f_back
        self._removeArgs()
        simulate(self._setArgs((sig,)))

    def current(self):
        if id(self) in current_cache:
            if current_cache[id(self)] is None:
                return self.value
            else:
                return current_cache[id(self)]
        current_cache[id(self)] = None
        if self.vec is not None:
            try:
                mask = 1 << self.vec.ls.index(self)
                val = bool(self.vec.current() & mask)
            except TypeError:
                val = self.args[0].current()
        else:
            val = self.args[0].current()

        self.checkEqual(val)
        current_cache[id(self)] = val
        return val

    def func(self, arg):
        return arg

class SyntheticSignal(Signal):
    def __init__(self, vec, idx):
        Signal.__init__(self)
        self.vec = vec
        self.idx = idx

    def getValue(self):
        try:
            value = self.vec.signalCurrent()
            if value != self.vec.incoming.value:
                print value, self.vec.incoming.value
            assert value == self.vec.incoming.value or isinstance(self.vec.incoming.value, DontCare)
            return bool(value & (1 << self.idx))
        except TypeError:
            return DontCare()

    value = property(getValue)

class Vector(object):
    def __init__(self, value, bits=16, enum=None):
        self.stack = inspect.currentframe().f_back
        self.func = None
        self.enum = enum
        if isinstance(value, Vector):
            self.__dict__ = value.__dict__
            return
        try:
            value = (s for s in value)
        except TypeError:
            if isinstance(value, bool):
                self.ls = [ConstantSignal(value)]
            elif isinstance(value, Signal):
                self.ls = [value]
            else:
                self.const = int(value) & (1 << bits) - 1
                self.ls = intToSignals(self.const, bits)

        else:
            self.ls = [s if isinstance(s, Signal) or isinstance(s, Vector) else ConstantSignal(s) for s in value]
        for s in self.ls:
            assert not isinstance(s, Vector) # don't combine vectors into new vector, use concat for that
            if s.vec is None:
                s.vec = self

    def makeEnum(self, enum):
        self.enum = enum
        return self

    def __repr__(self):
        if self.enum:
            return '{0}({1})'.format(type(self).__name__,
                                     ([k for k in self.enum if int(self.enum[k]) == int(self)] or ['?'])[0])
        else:
            try:
                val = signalsToInt(self.ls)
            except TypeError:
                val = DontCare()
            return '{0}({1}, {2})'.format(type(self).__name__, val, len(self))

    def __len__(self):
        return len(self.ls)

    def __getitem__(self, key):
        def current_func(start, stop):
            mask = ((1 << (stop - start)) - 1)
            def inner(val):
                return val >> start & mask

            return inner

        res = Vector(self.ls[key])
        assert (not isinstance(key, slice) or key.step is None)
        res.args = [self]
        res.func = current_func(
            (len(self.ls) + key if key < 0 else key) if not isinstance(key, slice) else
            0 if key.start is None else
            len(self.ls) + key.start if key.start < 0 else key.start,
            ((len(self.ls) + key if key < 0 else key) + 1) if not isinstance(key, slice) else
            len(self.ls) if key.stop is None else
            len(self.ls) + key.stop if key.stop < 0 else key.stop)
        return res

    def __iter__(self):
        return (self[n] for n in xrange(len(self)))

    def __add__(self, other):
        return SyntheticAdder(self, other)[:-1]

    def __sub__(self, other):
        return SyntheticSubtractor(self, other)[:-1]

    def __mul__(self, other):
        return SyntheticMultiplier(self, other)

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
        if len(other) == 1 and len(self) != 1:
            return Vector.op(__and__, self, other.dup(len(self)))
        elif len(self) == 1 and len(other) != 1:
            return Vector.op(__and__, self.dup(len(other)), other)
        else:
            return Vector.op(__and__, self, other)

    def __xor__(self, other):
        return Vector.op(__xor__, self, other)

    def __or__(self, other):
        return Vector.op(__or__, self, other)

    def __neg__(self):
        inv = ~self
        return KoggeStoneAdder(inv, Vector(0, len(self)), True)[0]

    def __pos__(self):
        return self

    def __abs__(self):
        return Vector(If(self[-1], -self, +self))

    def __invert__(self):
        res = Vector.op(__invert__, self)
        def current_func(val):
            return ~val & (1 << len(self) - 1)
        res.func = current_func
        return res

    def __int__(self):
        return signalsToInt(self.ls)

    def __lt__(self, other):
        return int(self) < int(other)

    def __le__(self, other):
        return int(self) <= int(other)

    def __ne__(self, other):
        return int(self) != int(other)

    def __gt__(self, other):
        return int(self) > int(other)

    def __ge__(self, other):
        return int(self) >= int(other)

    def __hash__(self):
        return int(self)

    def __nonzero__(self):
        return int(self) != 0

    def toUint(self):
        return signalsToInt(self.ls, False)

    def append(self, other):
        self.ls += other.ls if isinstance(other, Vector) else [other] if isinstance(other, Signal) else other[:]

    def concat(self, *others):
        def current_func(*vecs):
            sizes = [len(v.ls) if isinstance(v, Vector) else 1 for v in vecs]

            def inner(*vals):
                res = 0
                for i, val in enumerate(vals):
                    res |= val << sum(sizes[:i])
                return res

            return inner

        lss = [o.ls if isinstance(o, Vector) else [o] if isinstance(o, Signal) else o[:] for o in others]
        res = Vector(self.ls + sum(lss, []))
        res.args = [self] + list(others)
        res.func = current_func(self, *others)
        return res

    def extendBy(self, l, LSB=False, signed=False, signal=None):
        if isinstance(signal, Vector):
            if len(signal) != 1:
                raise TypeError('Vector given as signal argument is not of length 1')
        elif isinstance(signal, Signal):
            signal = Vector([signal])
        elif signal == None: signal = Vector(0, 1)
        if signed:
            signal = self[-1]
        if LSB:
            return signal.dup(l).concat(self)
        else:
            return self.concat(signal.dup(l))

    def extendTo(self, l, LSB=False, signed=False, signal=SFalse):
        return self.extendBy(l - len(self), LSB, signed, signal)

    def dup(self, n):
        def current_func(num):
            def inner(sigv):
                res = (1 << num) - 1 if (sigv & 1) else 0
                return res

            return inner

        if n == 1:
            return self
        res = Vector(self.ls[0:1] * n)
        res.args = [self]
        res.func = current_func(n)
        return res

    def signalCurrent(self):
        vals = [s.value for s in self.ls]
        try:
            val = sum((int(v) << i) for i, v in enumerate(vals))
        except TypeError:
            mask = sum(1 << i for i, v in enumerate(vals) if type(v) is DontCare)
            v = sum(1 << i for i, v in enumerate(vals) if type(v) is not DontCare and v)
            val = DontCare(v, mask)
        return val

    def current(self):
        if id(self) in current_cache:
            return current_cache[id(self)]
        try:
            return self.const
        except AttributeError:
            if self.func:
                args = [a.current() for a in self.args]
                try:
                    val = self.func(*args)
                except TypeError as e:
                    valTrue = self.func(*[a.val | a.mask if type(a) is DontCare else a for a in args])
                    valFalse = self.func(*[a.val & ~a.mask if type(a) is DontCare else a for a in args])
                    val = valTrue & (1 << len(self)) - 1 if valTrue == valFalse else \
                        DontCare(valTrue & valFalse, valTrue ^ valFalse)

            else:
                vals = [s.current() for s in self.ls]
                try:
                    val = sum((int(v) << i) for i, v in enumerate(vals))
                except TypeError:
                    mask = sum(1 << i for i, v in enumerate(vals) if type(v) is DontCare)
                    v = sum(1 << i for i, v in enumerate(vals) if type(v) is not DontCare and v)
                    val = DontCare(v, mask)
        current_cache[id(self)] = val
        self.checkEqual(val)
        return val

    def checkEqual(self, val):
        if type(val) is not DontCare and val & ((1 << len(self.ls)) - 1) != signalsToInt(self.ls, False):
            print self, "current result", val, "not equal to", signalsToInt(self.ls, False)
            stack = self.stack
            for _ in range(6):
                print 'File "{0}", line {1}, in {2}'.format(stack.f_code.co_filename, stack.f_lineno,
                                                            stack.f_code.co_name)
                stack = stack.f_back
                if not stack:
                    break
            if isinstance(self, FeedbackVector):
                return
            if not self.func:
                return
            try:
                print 'operation: File "{0}", line {1}, in {2}'.format(self.func.func_code.co_filename,
                                                                       self.func.func_code.co_firstlineno,
                                                                       self.func.__name__)
            except:
                print 'operation: <builtin> {0}'.format(self.func.__name__)
            print "on arguments:", self.args

    @classmethod
    # apply function pairwise on each signal in the argument vectors
    # fn is a function of multiple arguments that evaluates both on bool arguments
    # as on ints, treating them bitwise.
    def op(cls, fn, *args):
        res = Vector(fn(*arg) for arg in zip_all(*(a.ls for a in args)))
        res.func = fn
        res.args = args
        return res


    # op is an CircuitOper class that is applied to all signals of self. returns a 1-vector
    def reduce(self, op):
        obj = op(*self.ls)
        def current_func(arg):
            return obj.func(*intToBits(arg, len(self)))
        res = Vector(obj)
        res.func = current_func
        res.args = [self]
        return res


class ConstantVector(Vector):
    def __init__(self, value, bits=16, enum=None):
        def current_func():
            return value
        self.stack = inspect.currentframe().f_back
        self.func = current_func
        self.args = ()
        self.enum = enum
        self.const = int(value) & (1 << bits) - 1
        self.ls = intToSignals(self.const, bits)



def __eq__(self, other):
    return isinstance(other, Vector) and int(self) == int(other)


class TestVector(Vector):
    def __init__(self, value, bits=16):
        Vector.__init__(self, value)
        try:
            value = (s for s in value)
        except TypeError:
            self.ls = [value] if isinstance(value, TestSignal) else [TestSignal(b) for b in intToBits(int(value), bits)]
        else:
            self.ls = [s if isinstance(s, TestSignal) else TestSignal(s) for s in value]

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            signals = self.ls[key]
        if isinstance(value, int):
            bits = intToBits(value, len(signals))
        else:
            bits = value
        for b, s in zip(bits, signals):
            s.value = b.value if isinstance(b, Signal) else b
        self.const = value
        simulate(signals)


class Clock(TestVector):
    def __init__(self, initval=False):
        TestVector.__init__(self, initval, 1)
        self.registers = []

    def cycle(self, n=1, CPU_state=None):
        for _ in xrange(n):
            newvals = [(r, r.next.current()) for e, r in enumerate(self.registers)]
            self[:] = 1
            self[:] = 0
            current_cache.clear()
            for r, v in newvals:
                r.const = v
            for nn, (r, v) in enumerate(newvals):
                assert r.current() == v
                mask = (1 << len(r.ls)) - 1
                assert r.current() & mask == signalsToInt(r.ls) & mask, \
                    (r.current(), signalsToInt(r.ls), len(r.ls), _, nn)

            if CPU_state is not None:
                print "==============================="
                for k in CPU_state:
                    print k, ":", CPU_state[k]


class FeedbackVector(Vector):
    def __init__(self, value=None, bits=16):
        if value is None: value = 0

        if isinstance(value, Vector) and value.enum:
            enum = value.enum
        else:
            enum = None
        if isinstance(value, Vector):
            value = value.ls
        try:
            value = (FeedbackSignal(s) for s in value)
        except TypeError:
            value = [FeedbackSignal(b) for b in intToBits(int(value), bits)]
        else:
            value = [s if isinstance(s, FeedbackSignal) else FeedbackSignal(s) for s in value]
        Vector.__init__(self, value, bits, enum)

    def connect(self, vec):
        self.arg = vec
        for my_sig, other_sig in zip_all(self.ls, vec.ls):
            my_sig.connect(other_sig)

    def current(self):
        if id(self) in current_cache:
            if current_cache[id(self)] is None:
                return 0
            else:
                return current_cache[id(self)]
        current_cache[id(self)] = None

        val = self.arg.current()

        current_cache[id(self)] = val
        self.checkEqual(val)
        return val

class SyntheticIncomingSignal(Signal):
    def __init__(self, vec, *args):
        Signal.__init__(self)
        self.vec = vec
        argsigs = []
        for arg in args:
            if isinstance(arg, Vector):
                argsigs.extend(arg.ls)
            elif isinstance(arg, Signal):
                argsigs.append(arg)
        self._setArgs(argsigs)

    def eval(self):
        self.value = self.vec.signalCurrent()
        fanouts = {f for s in self.vec.ls for f in s.fanout}
        return fanouts


class SyntheticVector(Vector):
    def __init__(self, *args):
        self.args = args
        Vector.__init__(self, [SyntheticSignal(self, i) for i in xrange(len(self))])
        del self.func
        self.incoming = SyntheticIncomingSignal(self, *args)
        self.vec = None

    def signalCurrent(self):
        args = [a.signalCurrent() for a in self.args]
        try:
            val = self.func(*args)
        except TypeError as e:
            valTrue = self.func(*[a.val | a.mask if type(a) is DontCare else a for a in args])
            valFalse = self.func(*[a.val & ~a.mask if type(a) is DontCare else a for a in args])
            val = valTrue & (1 << len(self)) - 1 if valTrue == valFalse else \
                DontCare(valTrue & valFalse, valTrue ^ valFalse)
        return val

    def current(self):
        if id(self) in current_cache:
            return current_cache[id(self)]
        args = [a.current() for a in self.args]
        try:
            val = self.func(*args)
        except TypeError as e:
            valTrue = self.func(*[a.val | a.mask if type(a) is DontCare else a for a in args])
            valFalse = self.func(*[a.val & ~a.mask if type(a) is DontCare else a for a in args])
            val = valTrue & (1 << len(self)) - 1 if valTrue == valFalse else \
                DontCare(valTrue & valFalse, valTrue ^ valFalse)
        current_cache[id(self)] = val
        return val

    def check(self):
        if not self.vec:
            signals = self.build(*self.args)
            if isinstance(signals, Vector):
                self.vec = signals
            else:
                self.vec = Vector(signals)
        current_cache.clear()
        self.vec.checkEqual(self.current())

    def __len__(self):
        self.check()
        return len(self.signals)

class SyntheticAdder(SyntheticVector):
    def __len__(self):
        aLen = len(self.args[0])
        assert aLen == len(self.args[1])
        return aLen + 1

    def func(self, a, b, c=0):
        carry = ((a < 0) ^ (b < 0)) << (len(self) - 1)
        return a + b + c - carry

    def build(self, a, b, c=SFalse):
        return Vector.concat(*KoggeStoneAdder(a, b, c))

class SyntheticSubtractor(SyntheticVector):
    def __len__(self):
        aLen = len(self.args[0])
        assert aLen == len(self.args[1])
        return aLen + 1

    def func(self, a, b, c=0):
        borrow = ((a < 0) ^ (b >= 0)) << (len(self) - 1)
        return a - b - c - borrow

    def build(self, a, b, c=SFalse):
        return Vector.concat(*KoggeStoneAdder(a, ~b, ~c))


class SyntheticMultiplier(SyntheticVector):
    def __len__(self):
        return len(self.args[0]) + len(self.args[1])

    def func(self, a, b):
        return a * b

    def build(self, a, b):
        return Multiplier(a, b, True)

class SyntheticDecimalAdder(SyntheticVector):
    def __len__(self):
        aLen = len(self.args[0])
        assert aLen == len(self.args[1])
        return aLen + 1

    def func(self, a, b, c):
        ahex = ("%04x" % a).replace('f', '9').replace('e', '9').replace('d', '9').replace('c', '9').replace('b', '9').replace('a', '9')
        bhex = ("%04x" % b).replace('f', '9').replace('e', '9').replace('d', '9').replace('c', '9').replace('b', '9').replace('a', '9')
        res = int(ahex) + int(bhex) + c
        reshex = "%04u" % res
        return int(reshex, 16)

    def build(self, a, b, c):
        return DecimalAdder(a, b, c)


class DontCare(object):
    def __init__(self, val=0, mask=-1):
        self.val = val
        self.mask = mask

    def __eq__(self, other):
        return type(self) == type(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "DontCare"


class DontCareSignal(ConstantSignal):
    def __init__(self):
        ConstantSignal.__init__(self)
        self.value = DontCare()

    def __repr__(self):
        return '{0}()'.format(type(self).__name__)


DontCareSig = DontCareSignal()


class DontCareVector(Vector):
    def __init__(self, bits=16):
        super(DontCareVector, self).__init__([DontCareSig] * bits, bits)

    def __repr__(self):
        return "DontCareVector({0})".format(len(self.ls))


def intToBits(num, bits):
    return (((num >> i) & 1) == 1 for i in xrange(bits))


def intToSignals(num, bits):
    return [ConstantSignal(b) for b in intToBits(num, bits)]


def signalsToInt(ls, signed=True):
    try:
        num = sum(int(v.value) << i for i, v in enumerate(ls))
        if signed and len(ls) > 0 and ls[-1].value:
            num -= (1 << len(ls))
    except TypeError:
        mask = sum(1 << i for i, v in enumerate(ls) if type(v.value) is DontCare)
        num = sum(1 << i for i, v in enumerate(ls) if type(v.value) is not DontCare and v.value)
        return DontCare(num, mask)
    return num


class CircuitOper(Signal):
    def __init__(self, *args):
        Signal.__init__(self)
        self._setArgs(args)

    def partial(self, args, consts, D):
        cf = self.func(*consts)
        if len(args) == 1:
            res = self.constPartial(args[0], cf)
            args = res.simplifyTop(D)
        else:
            res = self.constPartial(self, cf)

        return res, args

    def simplify(self, D):
        args = self.simplifyTop(D)

        cvals = []
        vs = []
        for arg in args:
            if isinstance(arg, ConstantSignal):
                cvals.append(arg.value)
            else:
                vs.append(arg)
        sameArgs = tuple(a for a in args if type(a) == type(self) and a.fanout == [])

        if sameArgs:
            assert False
        elif cvals == []:
            res = self
        else:
            if vs == []:
                res = STrue if self.func(*cvals) else SFalse
                args = vs
            else:
                res, args = self.partial(vs, cvals, D)
        assert res.value == self.value
        return self.simplifyBottom(res, args, D)

    def oldSimplify(self, D):
        # print "simplify", self
        for f in self.fanout:
            assert self in f.args, f
        for a in self.args:
            if self in a.fanout:
                a.fanout.remove(self)
        self.args = [a.oldSimplify(D) for a in self.args]

        cs = []
        vs = []
        for arg in self.args:
            if type(arg) is ConstantSignal:
                cs.append(arg)
            else:
                vs.append(arg)

        sameArgs = [a for a in self.args if type(a) == type(self) and a.fanout == []]
        if sameArgs:
            res = sameArgs[0]
            for a in res.args:
                a.fanout.remove(res)
            res.args.extend(a for a in self.args if a != res)

        elif cs == []:
            res = self
        else:
            cvals = [c.value for c in cs]
            if vs == []:
                res = STrue if self.func(*cvals) else SFalse
            else:
                res = self.partial(vs, cvals)

        if res != self:
            for f in self.fanout:
                f.args[f.args.index(self)] = res
            res.fanout.extend(self.fanout)
            self.fanout = []
            self.args = []
        else:
            pass

        for a in res.args:
            # assert res not in a.fanout
            a.fanout.append(res)

        for f in res.fanout:
            assert res in f.args

        if sameArgs and cs:
            res = res.oldSimplify(D)

        D[self] = res
        return res

    def current(self):
        if id(self) in current_cache:
            return current_cache[id(self)]

        try:
            if self.vec is not None and self.vec.func and self not in self.vec.args:
                mask = 1 << self.vec.ls.index(self)
                val = self.vec.current()
                return bool(val & mask)
        except TypeError as e:
            return DontCare(1, 1) if val.mask & mask else bool(val.val & mask)

        try:
            args = []
            for arg in self.args:
                if arg.vec is not None and arg.vec.func:
                    try:
                        mask = 1 << arg.vec.ls.index(arg)
                        val = arg.vec.current()
                        args.append(bool(val & mask))
                    except TypeError:
                        args.append(arg.current())
                else:
                    args.append(arg.current())
            val = self.func(*args)
        except TypeError:
            valTrue = self.func(*[True if type(a) is DontCare else a for a in args])
            valFalse = self.func(*[False if type(a) is DontCare else a for a in args])
            if valTrue != valFalse:
                print 'Unequal !'
            val = valTrue if valTrue == valFalse else DontCare(valTrue ^ valFalse, not valTrue ^ valFalse)

        self.checkEqual(val)
        current_cache[id(self)] = val
        return val


class And(CircuitOper):
    def func(self, *a):
        return reduce(lambda p, q: p & q, a)

    @staticmethod
    def constPartial(v, cf):
        return v if cf else SFalse


class Or(CircuitOper):
    def func(self, *a):
        return reduce(lambda p, q: p | q, a)

    @staticmethod
    def constPartial(v, cf):
        return STrue if cf else v


class Xor(CircuitOper):
    def func(self, *a):
        return reduce(lambda p, q: p ^ q, a)

    @staticmethod
    def constPartial(v, cf):
        return Not(v) if cf else v


class Not(CircuitOper):
    def func(self, a):
        return not a


class Nor(CircuitOper):
    def func(self, a, b):
        return not (a | b)


class Enum(object):
    def __init__(self, *args):
        if len(args) == 1:
            try:
                kv = [(k, v) for v, k in args[0].iteritems()]
            except:
                kv = list(enumerate(args))
        else:
            kv = list(enumerate(args))
        bits = 0
        maxarg = max(k for k, _ in kv)
        while maxarg >= (1 << bits):
            bits += 1

        self._dict = {k: ConstantVector(v, bits).makeEnum(self) for v, k in kv}

    def __getitem__(self, item):
        return self._dict[item]

    def __getattr__(self, item):
        return self[item]

    def __iter__(self):
        return self._dict.__iter__()


def getDefault(val1, *vals):
    if isinstance(val1, Vector):
        for val in vals:
            if not isinstance(val, Vector):
                raise TypeError("Expected a %s but got a %s" % (Vector, type(val)))
            if not len(val) == len(val1):
                raise ValueError("Vectors are not of same size, {0} vs {1}".format(val, val1))
        return DontCareVector(len(val1))
        # return Vector(0, len(val1))
    elif isinstance(val1, Signal):
        for val in vals:
            if not isinstance(val, Signal):
                raise TypeError("Expected a %s but got a %s" % (Signal, type(val)))
        return DontCareSig
        # return SFalse
    elif isinstance(val1, dict):
        typ = type(val1)
        for val in vals:
            if not isinstance(val, dict):
                raise TypeError("Expected a %s but got a %s" % (type(val1), type(val)))
            if not len(val) == len(val1):
                raise ValueError("arguments are not of same size")
        return typ({k: getDefault(val1[k], *(val[k] for val in vals)) for k in val1})
    else:
        typ = type(val1)
        for val in vals:
            if not type(val) == type(val1):
                raise TypeError("Expected a %s but got a %s" % (type(val1), type(val)))
            if not len(val) == len(val1):
                raise ValueError("arguments are not of same size")
        zipped = zip_all(val1, *vals)
        return typ(getDefault(*v) for v in zipped)


def Case(state, cases, default=None):
    for k in cases.keys():
        if not (isinstance(k, Vector) or isinstance(k, Signal) or isinstance(k, basestring)):
            v = cases[k]
            for kk in k:
                cases[kk] = v
            del cases[k]
    if isinstance(state, Vector) and state.enum:
        cases = {state.enum[k] if isinstance(k, basestring) else k: cases[k] for k in cases}
        for k in cases:
            if k.enum != state.enum:
                raise ValueError("%s and %s are not of same enum type")
    zip_all(state, *cases.keys())
    length = len(state)
    if default is None:
        default = getDefault(*cases.values())
    else:
        getDefault(default, *cases.values())
    intcases = {k.toUint():cases[k] for k in cases}
    alts = [intcases.get(i, default) for i in xrange(2 ** length)]
    return Multiplexer(state, alts)


def TrueInCase(state, cases):
    if isinstance(cases, Vector):
        cases = [cases]
    new_cases = {}
    for k in cases:
        try:
            new_cases[k] = cases[k]
        except TypeError as e:
            new_cases[k] = Vector(True)
    first = new_cases.keys()[0]
    if isinstance(first, Vector) and first.enum is not None:
        allcases = set(first.enum[k] for k in first.enum)
        notcases = allcases - set(new_cases.keys())
        for k in notcases:
            new_cases[k] = Vector(False)
        return Case(state, new_cases)

    return Case(state, new_cases, default=Vector(False))


def HalfAdder(a, b):
    s = a ^ b
    c = a & b
    return s, c


def FullAdder(a, b, c_in):
    s1, c1 = HalfAdder(a, b)
    s2, c2 = HalfAdder(s1, c_in)
    c_out = c1 | c2
    return s2, c_out


def RippleCarryAdder(als, bls, c=SFalse):
    if isinstance(als, Vector): als = als.ls
    if isinstance(bls, Vector): bls = bls.ls
    assert len(als) == len(bls)
    assert isinstance(c, Signal)
    sls = Vector(0, 0)
    for a, b in zip_all(als, bls):
        s, c = FullAdder(a, b, c)
        sls.append(s)
    return sls, c


def KoggeStoneAdder(als, bls, c=False):
    if not isinstance(c, Vector):
        c = ConstantVector(bool(c), 1)
    assert isinstance(als, Vector)
    assert isinstance(bls, Vector)
    assert len(als) == len(bls)
    assert isinstance(c, Vector) and len(c) == 1
    prop = {0: als ^ bls}
    gen = {0: als & bls}

    step = 1
    while step < len(als):
        p = prop[step / 2]
        g = gen[step / 2]

        p_prev = p[:-step].extendTo(len(p), LSB=True)
        g_prev = g[:-step].extendTo(len(g), LSB=True, signal=c)
        prop[step] = p & p_prev
        gen[step] = (p & g_prev) | g
        step *= 2

    cls = Vector(c).concat(gen[step / 2])
    sls = Vector.op(lambda a, b, c: a ^ b ^ c, als, bls, cls[:-1])
    return sls, cls[-1]


def DecimalAdder(als, bls, c):
    if not isinstance(c, Vector):
        c = ConstantVector(bool(c), 1)
    assert isinstance(als, Vector)
    assert isinstance(bls, Vector)
    assert len(als) == len(bls)
    nibbles = [(als[i:i + 4].extendBy(1), bls[i:i + 4].extendBy(1)) for i in
               xrange(0, len(als), 4)]
    dst_out = []
    for src_n, dst_n in nibbles:
        sum_n, c_out = KoggeStoneAdder(src_n, dst_n, c)
        adjusted, adc_out = KoggeStoneAdder(sum_n, Vector(-10, 5))

        c = ~adjusted[-1]
        dst_out.append(If(c, adjusted[:-1], sum_n[:-1]))

    return Vector.concat(*dst_out + [c])


def Multiplier(als, bls, signed=True):
    if signed:
        als_sign = als[-1]
        bls_sign = bls[-1]
        als = If(als_sign, -als, als)
        bls = If(bls_sign, -bls, bls)

    sls, c = Vector(0, len(bls) - 1), SFalse
    for a in als:
        sls, c = RippleCarryAdder(a & bls, sls.concat(c))
        bls = bls.extendBy(1, LSB=True)

    res = sls.concat(c)
    if signed:
        sls_sign = als_sign ^ bls_sign
        res = If(sls_sign, -res, res)
    return res


def If(pred, cons, alt):
    def current_func():
        def inner(pred):
            return cons.current() if pred else alt.current()

        return inner

    assert isinstance(pred, Vector)
    npred = ~pred
    res = (pred & cons) | (npred & alt)
    res.args = [pred]
    res.func = current_func()
    return res


def SRLatch(s, r, init=False):
    q_fb = FeedbackSignal(bool(init))
    nq_fb = FeedbackSignal(not bool(init))
    nq = Nor(s, q_fb)
    q = Nor(r, nq_fb)
    q_fb.connect(q)
    nq_fb.connect(nq)
    return q, nq


def DLatch(d, e, init=False):
    return SRLatch(s=d & e, r=~d & e, init=init)


def DFlipFlop(d, clk, init=False):
    nclk = ~clk
    master = DLatch(d=d, e=nclk)
    slave = DLatch(d=master[0], e=clk, init=init)
    return slave


def Decoder(arg):
    def current_func():
        def inner(val):
            return 1 << val

        return inner

    a = arg[-1]
    not_a = ~a
    if len(arg) <= 1:
        res = not_a.concat(a)
    else:
        sub = Decoder(arg[:-1])
        res = (sub & not_a).concat(sub & a)
    res.args = [arg]
    res.func = current_func()
    return res


def Memory(clk, addr, data, isWrite, init=None):
    def current_func(values):
        def inner(mem_addr, data, isWrite):
            if isWrite:
                values[mem_addr] = data
            return values[mem_addr]

        return inner

    if not init:
        init = []
    initvalues = (init + [0] * ((1 << len(addr)) - len(init)))

    wordlines = Decoder(addr)  # [c & r for c in collines for r in rowlines]
    doWrite = isWrite & ~clk
    isWriteVec = doWrite.dup(len(data))
    sets = Vector.op(__and__, isWriteVec, data)
    resets = Vector.op(__and__, isWriteVec, ~data)

    q_outs = [[SRLatch(access & s, access & r, b)[0] & access
               for (s, r, b) in zip_all(sets.ls, resets.ls, list(intToBits(val, len(data))))]
              for access, val in zip_all(wordlines.ls, initvalues)]
    res = Vector([Or(*l) for l in zip(*q_outs)])

    res.args = [addr, data, isWrite]
    res.func = current_func(initvalues)
    res.q_outs = q_outs
    return res


class SyntheticMemory(SyntheticVector):
    def __init__(self, clk, addr, data, isWrite, init=None):
        SyntheticVector.__init__(self, clk, addr, data, isWrite, init)
        if not init:
            init = []
        self.values = (init + [0] * ((1 << len(addr)) - len(init)))

    def __len__(self):
        clk, addr, data, isWrite, init = self.args
        return len(data)

    def func(self, clk, addr, data, isWrite, init=None):
        if isWrite:
            self.values[addr] = data
        return self.values[addr]

    def build(self, clk, addr, data, isWrite, init=None):
        return Memory(clk, addr, data, isWrite, self.values)


def ROM(_, addr, data, __, init):
    def current_func(values, q_outs):
        def inner(mem_addr):
            q_vals = [signalsToInt([q.args[0] for q in qs], False) for qs in q_outs]
            uvalues = [v & ~(-1 << len(q_outs[0])) for v in values]
            if q_vals != uvalues:
                assert q_vals == uvalues, (q_vals, uvalues)
            return values[mem_addr]

        return inner

    if not init:
        init = []
    delayedAddr = addr

    wordlines = Decoder(delayedAddr)  # [c & r for c in collines for r in rowlines]

    initvalues = (init + [0] * (len(wordlines) - len(init)))

    q_outs = [[ConstantSignal(b) & access for b in list(intToBits(val, len(data)))]
              for access, val in zip_all(wordlines.ls, initvalues)]
    res = Vector([Or(*l) for l in zip(*q_outs)])
    res.args = [delayedAddr]
    res.func = current_func(initvalues, q_outs)
    res.q_outs = q_outs
    return res


def Multiplexer(sel, alts):
    enables = Decoder(sel)

    def current_func(alts):
        def inner(sel):
            return alts[sel].current()

        return inner

    def combiner(alts):
        alts = list(alts)
        alt0 = alts.ls[0] if isinstance(alts, Vector) else alts[0]
        if isinstance(alt0, Signal):
            for alt in alts[1:]:
                if not isinstance(alt, Signal):
                    raise TypeError()
            return Or(*(Vector(alts) & enables).ls)
        elif isinstance(alt0, dict):
            typ = type(alt0)
            keys = set(alt0.keys())
            for alt in alts[1:]:
                if type(alt) is not typ:
                    raise TypeError()
                if set(alt.keys()) != keys:
                    raise ValueError()
            return typ({k: combiner(a[k] for a in alts) for k in alt0})

        elif isinstance(alt0, Vector):
            ln = len(alt0)
            for alt in alts[1:]:
                if not isinstance(alt, Vector):
                    raise TypeError()
                if len(alt) is not ln:
                    raise ValueError()
            res = Vector(combiner(a.ls[i] for a in alts) for i in xrange(ln))
            res.args = [sel]
            res.func = current_func(alts)
            return res
        else:
            typ = type(alt0)
            ln = len(alt0)
            for alt in alts[1:]:
                if type(alt) is not typ:
                    raise TypeError()
                if len(alt) is not ln:
                    raise ValueError()
            return typ(combiner(a[i] for a in alts) for i in xrange(ln))

    return combiner(alts)


def calcAreaDelay(inputs):
    if isinstance(inputs, Vector):
        inputs = inputs.ls
    levels = [set(inputs)]
    totals = set()
    i = 0
    while len(levels[i]) > 0:
        levels.append(set(c
                          for inp in levels[i]
                          for c in inp.fanout
                          if not isinstance(c, FeedbackSignal)))
        totals.update(levels[i])
        fbs = set(c
                          for inp in levels[i]
                          for c in inp.fanout
                          if isinstance(c, FeedbackSignal))
        levels[i].update(set(f for c in fbs for f in c.fanout) - totals)
        i += 1
    return len(totals), i - 1


def FlipFlops(dls, clk, init=None):
    if init is None: init = [None] * len(dls)
    outs = zip_all(*tuple(DFlipFlop(d, clk, i) for d, i in zip(dls, init)))
    if len(outs) == 0:
        return Vector([])
    return Vector(outs[0])


class Register(FeedbackVector):
    def __init__(self, clk, value=None, bits=16):
        def inner():
            return self.current()

        FeedbackVector.__init__(self, value, bits)
        self.clk = clk
        self.const = signalsToInt(self.ls, False)
        self.next = None
        self.prev = self
        clk.registers.append(self)
        self.func = inner
        self.args = []

    def connect(self, next):
        self.next = next
        flops = FlipFlops(next.ls, self.clk.ls[0], self.ls)
        FeedbackVector.connect(self, flops)
        self.checkEqual(self.const)
        return self

    def ccase(self, state, cases, default=None):
        if default is None: default = self
        self.connect(Case(state, cases, default))

    def current(self):
        val = self.const

        self.checkEqual(val)
        return val
