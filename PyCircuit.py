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
                continue
        next_signals.update(sig.eval())
    if len(next_signals) > 0:
        over_limit = False
        for e in (e for e in recur if recur[e] >= 16):
            over_limit = True
            print "After sim_step %i: FeedbackSignal %i (%i times)" % (sim_steps, id(e), recur[e])
            fr = e.stack
            for _ in xrange(4):
                print 'File "%s", line %i, in %s\n%s' % (
                    fr.f_code.co_filename, fr.f_lineno, fr.f_code.co_name, inspect.findsource(fr)[0][fr.f_lineno - 1]),
                fr = fr.f_back

            frame = inspect.currentframe()
            cs = set(e)
            while True:
                if e in cs:
                    break
                print cs
                frame = frame.f_back
                cs = set(s for s in frame.f_locals['signals'] if cs & set(s.fanout))
                if e in cs:
                    break

        if over_limit:
            raise Exception("Too many iterations")
        simulate(next_signals, recur)
    else:
        if len(recur) > 0:
            simulate(s for s in recur if recur[s] >= 4)

def simplify(*args):
    """
    Replaces circuits by simpler, cheaper variants wherever it can by performing
    constant propagation, and replacing chained binary circuits by n-ary ones
    Returns a tuple with replacements of the input arguments
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
                res = (1 << num) - 1 if sigv else 0
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

class TestSignal(Signal):
    def set(self, value=True):
        self.set_value = value.value if isinstance(value, Signal) else value
        self.simulate()

    def reset(self):
        self.set(False)

    def func(self):
        return self.set_value


class ClockSignal(TestSignal):
    def cycle(self, n=1, CPU_state=None):
        for _ in xrange(n):
            self.set()
            self.reset()
            current_cache.clear()
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

    @staticmethod
    def func(val):
        return val

class Vector(object):
    def __init__(self, value, bits=16, enum=None):
        self.func = None
        self.enum = enum
        try:
            value = (s for s in value)
        except TypeError:
            if isinstance(value, Signal):
                self.ls = [value]
            else:
                self.const = int(value) & (1 << bits) - 1
                self.ls = intToSignals(self.const, bits)

        else:
            self.ls = [s if isinstance(s, Signal) else Signal(s) for s in value]
        for s in self.ls:
            s.vec = self

    def __repr__(self):
        if self.enum:
            return 'Vector(%s)' % (tuple(k for k in self.enum if self.enum[k] == self))
        else:
            try:
                val = signalsToInt(self.ls)
            except TypeError:
                val = DontCare()
            return 'Vector({0}, {1})'.format(val, len(self))

    def __len__(self):
        return len(self.ls)

    def __getitem__(self, key):
        def current_func(start, stop):
            def inner(val):
                return val >> start & ((1 << (stop - start)) - 1)

            return inner
        if isinstance(key, slice):
            res = Vector(self.ls[key])
            assert (key.step is None)
            res.args = [self]
            res.func = current_func(
                0 if key.start is None else len(self.ls) + key.start if key.start < 0 else key.start,
                len(self.ls) if key.stop is None else len(self.ls) + key.stop if key.stop < 0 else key.stop)
            return res
        else:
            return self.ls[key]

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

    # concatenate with list
    # def __radd__(self, other):
    #     return Vector(other + self[:])


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
        if signed:
            signal = self.ls[-1]
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
                    val = valTrue if valTrue == valFalse else DontCare(valTrue ^ valFalse, not valTrue ^ valFalse)
                    val = val & ((1 << len(self)) - 1)

            else:
                vals = [s.current() for s in self.ls]
                try:
                    val = sum((int(v) << i) for i, v in enumerate(vals))
                except TypeError:
                    mask = sum(1 << i for i, v in enumerate(vals) if type(v) is DontCare)
                    v = sum(1 << i for i, v in enumerate(vals) if type(v) is not DontCare and v)
                    val = DontCare(v, mask)
        current_cache[id(self)] = val
        # if val & ((1 << len(self.ls)) - 1) != signalsToInt(self.ls, False):
        #     print self, "current result", val, "not equal to", signalsToInt(self.ls, False)
        return val

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
    def __init__(self, value=None, bits=16, state=None):
        if value is None: value = 0
        super(FeedbackVector, self).__init__(value)
        if isinstance(value, Vector) and value.enum:
            self.enum = value.enum
        else:
            self.enum = state
        try:
            value = (s for s in value)
        except TypeError:
            self.ls = [FeedbackSignal(b) for b in intToBits(int(value), bits)]
        else:
            self.ls = [s if isinstance(s, FeedbackSignal) else FeedbackSignal(s) for s in value]

    def __getitem__(self, key):
        return Vector.__getitem__(self, key)

    def connect(self, vec):
        for my_sig, other_sig in zip_all(self.ls, vec.ls):
            my_sig.connect(other_sig)


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
        self.ls = [DontCareSig] * bits
        self.func = None

    def __repr__(self):
        return "DontCareVector({0})".format(len(self.ls))

class GuardedEval():
    def __init__(self, sig, name, vect):
        self.sig = sig
        self.name = name
        self.vect = vect

    def __call__(self, *args, **kwargs):
        oldval = self.sig.value
        res = Signal.eval(self.sig)
        val = self.sig.value
        if oldval != val:
            print self.name, "changed at sim_step %i to %s" % (sim_steps, self.vect)
        return res


class GuardedVector(Vector):
    def __init__(self, value, bits=16, enum=None):
        Vector.__init__(self, value, bits, enum)
        stack = inspect.currentframe().f_back
        name = ""
        while True:
            var = [k for k in stack.f_locals if stack.f_locals[k] is value]
            if not var:
                name = name[1:]
                break
            fn = stack.f_code.co_name
            name += "." + var[0] + ":" + fn
            stack = stack.f_back

        for s in self.ls:
            if isinstance(s, Signal):
                s.eval = GuardedEval(s, name, self)
            else:
                print "unknown thing in GuardedVector:", s


def EnumVector(enum, value, bits=16):
    return Vector(value, bits, enum=enum)


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
        mask = sum(1 << i for i, v in enumerate(ls) if type(v) is DontCareSignal)
        num = sum(1 << i for i, v in enumerate(ls) if type(v) is not DontCareSignal and v.value)
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
            args = []
            for arg in self.args:
                if arg.vec and arg.vec.func:
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
            valTrue = self.func(*[True if a is DontCare else a for a in args])
            valFalse = self.func(*[False if a is DontCare else a for a in args])
            val = valTrue if valTrue == valFalse else DontCare

        if val != self.value:
            print self, "current result", val, "not equal to", self.value
        current_cache[id(self)] = val
        return val


class And(CircuitOper):
    def func(self, *a):
        return reduce(lambda p, q: p & q, a)

    def constPartial(self, v, cf):
        return v if cf else Signal(False)


class Or(CircuitOper):
    def func(self, *a):
        return reduce(lambda p, q: p | q, a)

    def constPartial(self, v, cf):
        return Signal(True) if cf else v


class Xor(CircuitOper):
    def func(self, *a):
        return reduce(lambda p, q: p ^ q, a)

    def constPartial(self, v, cf):
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
            new_cases[k] = Signal(True)
    first = new_cases.keys()[0]
    if isinstance(first, Vector) and first.enum != None:
        allcases = set(first.enum[k] for k in first.enum)
        notcases = allcases - set(new_cases.keys())
        for k in notcases:
            new_cases[k] = Signal(False)
        return Case(state, new_cases)

    return Case(state, new_cases, default=Signal(False))


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
    if isinstance(pred, Vector):
        pred = Or(*pred[:])
    npred = Not(pred)
    res = (pred & cons) | (npred & alt)
    return res


def SRLatch(s, r, init=False):
    q_fb = FeedbackSignal(init)
    nq_fb = FeedbackSignal(not init)
    nq = Nor(s, q_fb)
    q = Nor(r, nq_fb)
    q_fb.connect(q)
    nq_fb.connect(nq)
    return q, nq


def DLatch(d, e):
    return SRLatch(s=d & e, r=Not(d) & e)


def DFlipFlop(d, clk):
    nclk = Not(clk)
    master = DLatch(d=d, e=nclk)
    slave = DLatch(d=master[0], e=clk)
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


def GuardedMemory(mem_addr, data, isWrite, init=None):
    g_mem_addr = GuardedVector(mem_addr)
    g_data = GuardedVector(data)
    g_isWrite = GuardedVector(isWrite)
    out = Memory(g_mem_addr, g_data, g_isWrite[0], init)
    return out


def Memory(mem_addr, data, isWrite, init=None):
    if not init:
        init = []
    collines = Decoder(mem_addr[len(mem_addr) / 2:])
    rowlines = Decoder(mem_addr[:len(mem_addr) / 2])

    wordlines = [c & r for c in collines for r in rowlines]

    sr = [(isWrite & d, isWrite & ~d) for d in data]
    initvalues = (init + [0] * (len(wordlines) - len(init)))

    q_outs = [[SRLatch(access & s, access & r, b)[0] & access
               for ((s, r), b) in zip_all(sr, list(intToBits(val, len(sr))))]
              for access, val in zip_all(wordlines, initvalues)]
    return Vector([Or(*l) for l in zip(*q_outs)])


def Multiplexer(sel, alts):
    enables = Decoder(sel)

    def current_func(alts):
        def inner(sel):
            return alts[sel].current()

        return inner

    def combiner(alts):
        alts = list(alts)
        alt0 = alts[0]
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
            res = Vector(combiner(a[i] for a in alts) for i in xrange(ln))
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


def FlipFlops(dls, clk):
    res = zip_all(*tuple(DFlipFlop(d, clk) for d in dls))
    if len(res) == 0:
        return Vector([])
    return Vector(res[0])


class Register(FeedbackVector):
    def __init__(self, clk, value=None, bits=16, state=None):
        self.clk = clk
        super(Register, self).__init__(value, bits, state)
        self.next = None
        self.prev = self

    def connect(self, next):
        self.next = next
        FeedbackVector.connect(self, FlipFlops(next, self.clk))

    def ccase(self, state, cases, default=None):
        if default is None: default = self
        self.connect(Case(state, cases, default))