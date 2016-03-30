'''
Created on Feb 4, 2014

@author: leone
'''
import inspect
from collections import Counter
from operator import __and__, __xor__, __or__, __invert__

sim_steps = 0


def zip_all(*args):
    for i in xrange(0, len(args)):
        for j in xrange(i + 1, len(args)):
            if len(args[i]) != len(args[j]):
                raise TypeError('Vectors are not of equal size')
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
    do_recur = False
    if recur is None:
        recur = Counter()
        do_recur = True
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
                print "findPrev", sig
                nexts = sum((sig.eval() for sig in path), [])
                # recur[sig] = 0

                new_nexts = set(nexts)
                next_signals.update(new_nexts)

                continue
        next_signals.update(sig.eval())
    global monitored
    if any(s in signals for s in monitored):
        printMonitored()
    if not next_signals:
        return
    simulate(next_signals, recur)

    # if do_recur:
    #     # for s in recur:
    #     #     if recur[s] >= 5:
    #     #         s.simulate()
    #     simulate(s for s in recur)
    #     simulate(signals)

    return


def simplify(*args):
    """
    Replaces circuits by simpler, cheaper variants wherever it can by performing
    constant propagation, and replacing chained binary circuits by n-ary ones
    Returns a tuple with replacements of the input arguments
    :param args: all the signals to simplify
    """
    new_args = []
    for arg in args:
        if isinstance(arg, Vector):
            arg = Vector(a.simplify() for a in arg)
        elif isinstance(arg, Signal):
            arg = arg.simplify()
        else:
            arg = [a.simplify() for a in arg]
        new_args.append(arg)
    return tuple(new_args)


current_cache = {}


class Signal(object):
    def __init__(self, initval=False):
        self.args = []
        self.value = initval.value if isinstance(initval, Signal) else bool(initval)
        self.fanout = []
        self.vec = None

    def simplify(self):
        return self

    def __xor__(self, other):
        return Xor(self, other)

    def __or__(self, other):
        return Or(self, other)

    def __and__(self, other):
        if isinstance(other, Vector):
            return Vector.op(__and__, self * len(other), other)
        else:
            return And(self, other)

    def __invert__(self):
        return Not(self)

    def __mul__(self, other):
        def current_func(num):
            def inner(sigv):
                res = (sigv << num) - 1 if sigv else 0
                return res

            return inner

        if other == 1:
            return self
        res = Vector([self] * other)
        res.args = [self]
        res.func = current_func(other)
        return res

    def __repr__(self):
        return '{0}({1}) [#{2:X}]'.format(type(self).__name__, self.value, id(self))

    def __len__(self):
        return 1

    def __iter__(self):
        return iter([self])

    def concat(self, *others):
        return Vector(self).concat(*others)

    def func(self, *args):
        raise NotImplementedError()

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

    def current(self):
        return self.value

    def checkEqual(self, val):
        if val != self.value:
            print self, "current result", val, "not equal to", self.value


class TestSignal(Signal):
    def set(self, value=True):
        self.set_value = value.value if isinstance(value, Signal) else value
        self.simulate()

    def reset(self):
        self.set(False)

    def func(self):
        return self.set_value


class ClockSignal(TestSignal):
    def __init__(self, initval=False):
        TestSignal.__init__(self, initval)
        self.registers = []

    def cycle(self, n=1, CPU_state=None):
        for _ in xrange(n):
            newvals = [(r, r.next.current()) for e, r in enumerate(self.registers)]
            self.set()
            self.reset()
            current_cache.clear()
            for r, v in newvals:
                r.const = v
            for nn, (r, v) in enumerate(newvals):
                assert r.current() == v
                mask = (1 << len(r.ls)) - 1
                assert r.current() & mask == signalsToInt(r.ls) & mask, (
                r.current(), signalsToInt(r.ls), len(r.ls), _, nn)

            if CPU_state is not None:
                print "==============================="
                for k in CPU_state:
                    print k, ":", CPU_state[k]


class FeedbackSignal(Signal):
    def connect(self, sig):
        self.stack = inspect.currentframe().f_back
        self.args = [sig]
        sig.fanout.append(self)
        self.simulate()

    def current(self):
        if id(self) in current_cache:
            if current_cache[id(self)] is None:
                return 0
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

    @staticmethod
    def func(val):
        return val


class Vector(object):
    def __init__(self, value, bits=16, enum=None):
        self.stack = inspect.currentframe().f_back
        self.func = None
        self.enum = enum
        try:
            value = (s for s in value)
        except TypeError:
            if isinstance(value, bool):
                self.ls = [Signal(value)]
            elif isinstance(value, Signal):
                self.ls = [value]
            else:
                self.const = int(value) & (1 << bits) - 1
                self.ls = intToSignals(self.const, bits)

        else:
            self.ls = [s if isinstance(s, Signal) else Signal(s) for s in value]
        for s in self.ls:
            if s.vec is None:
                s.vec = self

    def __repr__(self):
        if self.enum:
            return '{0}({1})'.format(type(self).__name__, [k for k in self.enum if self.enum[k] == self][0])
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
            def inner(val):
                return val >> start & ((1 << (stop - start)) - 1)

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
        return iter(self.ls)

    def __add__(self, other):
        return KoggeStoneAdder(self, other)[0]

    def __sub__(self, other):
        return KoggeStoneAdder(self, ~other, Signal(1))[0]

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
        if isinstance(other, Signal):
            return Vector.op(__and__, self, other * len(self))
        elif len(other) == 1:
            return Vector.op(__and__, self, other.ls[0] * len(self))
        elif len(self) == 1:
            return Vector.op(__and__, self.ls[0] * len(other), other)
        else:
            return Vector.op(__and__, self, other)

    def __xor__(self, other):
        return Vector.op(__xor__, self, other)

    def __or__(self, other):
        return Vector.op(__or__, self, other)

    def __neg__(self):
        inv = ~self
        return inv + Vector(1, len(self))

    def __pos__(self):
        return self

    def __abs__(self):
        return Vector(If(self[-1], -self, +self))

    def __invert__(self):
        return Vector.op(__invert__, self)

    def __int__(self):
        return signalsToInt(self.ls)

    def __lt__(self, other):
        return int(self) < int(other)

    def __le__(self, other):
        return int(self) <= int(other)

    def __eq__(self, other):
        return isinstance(other, Vector) and int(self) == int(other)

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

    def extendBy(self, l, LSB=False, signed=False, signal=Signal(0)):
        if isinstance(signal, Vector):
            if len(signal) == 1:
                signal = signal.ls[0]
            else:
                raise TypeError('Vector given as signal argument is not of length 1')
        if signed:
            signal = self.ls[-1]
        if isinstance(signal, Vector):
            signal = signal.ls[0]
        if LSB:
            return (signal * l).concat(self)
        else:
            return self.concat(signal * l)

    def extendTo(self, l, LSB=False, signed=False, signal=Signal(0)):
        return self.extendBy(l - len(self), LSB, signed, signal)

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
            for _ in range(4):
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
    def op(cls, fn, *args):
        res = Vector(fn(*a) for a in zip_all(*args))
        res.func = fn
        res.args = args
        return res


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
            s.set_value = b.value if isinstance(b, Signal) else b
        self.const = value
        simulate(signals)


class FeedbackVector(Vector):
    def __init__(self, value=None, bits=16):
        if value is None: value = 0

        if isinstance(value, Vector) and value.enum:
            enum = value.enum
        else:
            enum = None
        try:
            value = (s for s in value)
        except TypeError:
            value = [FeedbackSignal(b) for b in intToBits(int(value), bits)]
        else:
            value = [s if isinstance(s, FeedbackSignal) else FeedbackSignal(s) for s in value]
        super(FeedbackVector, self).__init__(value, bits, enum)

    def __getitem__(self, key):
        return Vector.__getitem__(self, key)

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


class DontCareSignal(Signal):
    def __init__(self):
        super(DontCareSignal, self).__init__()
        self.value = DontCare()

    def __repr__(self):
        return '{0}()'.format(type(self).__name__)


DontCareSig = DontCareSignal()


class DontCareVector(Vector):
    def __init__(self, bits=16):
        super(DontCareVector, self).__init__([DontCareSig] * bits, bits)

    def __repr__(self):
        return "DontCareVector({0})".format(len(self.ls))


def EnumVector(enum, value, bits=16):
    res = Vector(value, bits, enum=enum)
    res.func = lambda x: x
    res.args = [value]
    return res


def intToBits(num, bits):
    return (((num >> i) & 1) == 1 for i in xrange(bits))


def intToSignals(num, bits):
    return [Signal(b) for b in intToBits(num, bits)]


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
        self.args = list(args)

        for arg in args:
            arg.fanout.append(self)
        self.simulate()

    def partial(self, args, consts):
        cf = self.func(*consts)
        if len(args) == 1:
            v = args[0]
        else:
            self.args = args
            v = self

        return self.constPartial(v, cf)

    def simplify(self):
        self.args = [a.simplify() for a in self.args]
        for f in self.fanout:
            assert self in f.args
        for a in self.args:
            assert self in a.fanout
            a.fanout.remove(self)

        cs = []
        vs = []
        for arg in self.args:
            if type(arg) is Signal:
                cs.append(arg)
            else:
                vs.append(arg)

        sameArgs = [a for a in self.args if type(a) == type(self) and a.fanout == []]
        if sameArgs:
            res = sameArgs[0]
            res.args.extend(a for a in self.args if a != res)

        elif cs == []:
            res = self
        else:
            cvals = [c.value for c in cs]
            if vs == []:
                res = Signal(self.func(*cvals))
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
            a.fanout.append(res)

        for f in res.fanout:
            assert res in f.args

        if sameArgs and cs:
            return res.simplify()
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
        return v if cf else Signal(False)


class Or(CircuitOper):
    def func(self, *a):
        return reduce(lambda p, q: p | q, a)

    @staticmethod
    def constPartial(v, cf):
        return Signal(True) if cf else v


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

        self._dict = {k: EnumVector(self, v, bits) for v, k in kv}

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
        # return Signal()
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
    alts = [cases.get(Vector(i, length), default) for i in xrange(2 ** length)]
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


def RippleCarryAdder(als, bls, c=Signal()):
    sls = Vector(0, 0)
    for a, b in zip_all(als, bls):
        s, c = FullAdder(a, b, c)
        sls.append(s)
    return sls, c


def KoggeStoneAdder(als, bls, c=Signal()):
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


def DecimalAdder(src, dst_in, c_in):
    nibbles = [(src[i:i + 4].extendBy(1), dst_in[i:i + 4].extendBy(1)) for i in
               xrange(0, len(src), 4)]
    dst_out = []
    for src_n, dst_n in nibbles:
        sum_n, c_out = KoggeStoneAdder(src_n, dst_n, c_in)
        adjusted, adc_out = KoggeStoneAdder(sum_n, Vector(-10, 5))

        c_in = ~adjusted[-1]
        dst_out.append(If(c_in, adjusted[:-1], sum_n[:-1]))

    return Vector(0, 0).concat(*dst_out), c_in


def Multiplier(als, bls, signed=True):
    if signed:
        als_sign = als[-1]
        bls_sign = bls[-1]
        als = If(als_sign, -als, als)
        bls = If(bls_sign, -bls, bls)

    sls, c = Vector(0, len(bls) - 1), Signal()
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

    if isinstance(pred, Vector):
        pred = Or(*pred[:])
    npred = Not(pred)
    res = (pred & cons) | (npred & alt)
    res.args = [pred]
    res.func = current_func()
    return res


def SRLatch(s, r, init=False):
    q_fb = FeedbackSignal(init)
    nq_fb = FeedbackSignal(not init)
    nq = Nor(s, q_fb)
    q = Nor(r, nq_fb)
    q_fb.connect(q)
    nq_fb.connect(nq)
    return q, nq


def DLatch(d, e, init=False):
    return SRLatch(s=d & e, r=Not(d) & e, init=init)


def DFlipFlop(d, clk, init=False):
    nclk = Not(clk)
    master = DLatch(d=d, e=nclk)
    slave = DLatch(d=master[0], e=clk, init=init)
    return slave


def Decoder(arg):
    def current_func():
        def inner(val):
            return 1 << val

        return inner

    a = arg.ls[-1]
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
    def current_func(values, q_outs):
        def inner(mem_addr, data, isWrite):
            if isWrite:
                values[mem_addr] = data
            q_vals = [signalsToInt([q.args[0] for q in qs], False) for qs in q_outs]
            uvalues = [v & ~(-1 << len(q_outs[0])) for v in values]
            if q_vals != uvalues:
                assert q_vals == uvalues, (q_vals, uvalues)
            return values[mem_addr]

        return inner

    if not init:
        init = []
    delayedAddr = addr
    collines = Decoder(delayedAddr[len(delayedAddr) / 2:])
    rowlines = Decoder(delayedAddr[:len(delayedAddr) / 2])

    wordlines = [c & r for c in collines for r in rowlines]

    delayedWrite = ~~(isWrite & clk)
    doWrite = isWrite & ~clk

    isWriteVec = doWrite.extendTo(len(data), signal=doWrite)
    sets = Vector.op(__and__, isWriteVec, data)
    resets = Vector.op(__and__, isWriteVec, ~data)
    initvalues = (init + [0] * (len(wordlines) - len(init)))

    q_outs = [[SRLatch(access & s, access & r, b)[0] & access
               for (s, r, b) in zip_all(sets, resets, list(intToBits(val, len(data))))]
              for access, val in zip_all(wordlines, initvalues)]
    res = Vector([Or(*l) for l in zip(*q_outs)])
    res.args = [delayedAddr, data, isWrite]
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
            return Or(*(Vector(alts) & enables))
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


def RegisterFile(addr1, addr2, addr_w, data_w, clk_w):
    wordlines_w = Decoder(addr_w)

    q_outs = [Vector(DFlipFlop(d, mem_wr & clk_w)[0] for d in data_w) for mem_wr in wordlines_w]

    data1 = Multiplexer(addr1, q_outs)
    data2 = Multiplexer(addr2, q_outs)
    return data1, data2


def calcAreaDelay(inputs):
    if isinstance(inputs, Vector):
        inputs = inputs.ls
    levels = [set(inputs)]
    i = 0
    while len(levels[i]) > 0:
        levels.append(set(c
                          for inp in levels[i]
                          for c in inp.fanout
                          if not isinstance(c, FeedbackSignal)))

        i += 1
    return len(set().union(*levels[1:])), i - 1


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

        super(Register, self).__init__(value, bits)
        self.clk = clk
        self.const = signalsToInt(self.ls, False)
        self.next = None
        self.prev = self
        clk.registers.append(self)
        self.func = inner
        self.args = []

    def connect(self, next):
        self.next = next
        FeedbackVector.connect(self, FlipFlops(next, self.clk, [e.value for e in self.ls]))
        self.checkEqual(self.const)
        return self

    def ccase(self, state, cases, default=None):
        if default is None: default = self
        self.connect(Case(state, cases, default))

    def current(self):
        val = self.const

        self.checkEqual(val)
        return val
