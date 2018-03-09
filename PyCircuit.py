'''
Created on Feb 4, 2014

@author: leone
'''
import inspect
import itertools
from functools import reduce
from operator import __and__, __xor__, __or__, __invert__, or_, and_, xor, eq, ne

sim_steps = 0


def len_all(*args):
    if len(args) == 0:
        return 0
    for i in range(0, len(args)):
        for j in range(i + 1, len(args)):
            if len(args[i]) != len(args[j]):
                raise TypeError('Vectors are not of equal size: %i and %i' % (len(args[i]), len(args[j])))
    return len(args[i])


def zip_all(*args):
    len_all(*args)
    return list(zip(*args))


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
    print("{:>8d}:  {}".format(sim_steps,
                               "".join(("{:^16}".format([k for k in v.enum if int(v.enum[k]) == int(v)])
                                        if v.enum else "{:^18}".format(v.toUint()))
                                       if isinstance(v, Vector) else
                                       "".join(
                                           "D" if isinstance(s.value, DontCare) else "-" if s.value else "_" for s in v)
                                       for v in monitoredVectors)))


def simulate(signals):
    global sim_steps
    sim_steps += 1
    next_signals = set()
    for sig in signals:
        if isinstance(sig, FeedbackSignal):
            next_signals.update(*(sig.eval() for sig in sig._path))

        next_signals.update(sig.eval())
    global monitored
    if any(s in signals for s in monitored):
        printMonitored()
    if not next_signals:
        return
    simulate(next_signals)


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
        DT = D[True]
        del D[True]
        DF = D[False]
        del D[False]
        DN = D[None]
        del D[None]
    vecs = set([k.vec for k in D if hasattr(k, 'vec')])
    for v in vecs:
        if v is None: continue
        val = v.current()
        v.ls = [D[e] if e in D else e for e in v.ls]
        v.checkEqual(val)
    return tuple(new_args)


current_cache = {}


def intToBits(num, bits):
    return (((num >> i) & 1) for i in range(bits))


def intToSignals(num, bits):
    return [ConstantSignal(b) for b in intToBits(num, bits)]


def DontCareToBits(dc, bits):
    return (DontCareBool if ((dc.mask >> i) & 1) == 1 else ((dc.val >> i) & 1) == 1 for i in range(bits))


def signalsToInt(ls, signed=True):
    try:
        num = sum(int(v.value) << i for i, v in enumerate(ls))
        if signed and len(ls) > 0 and ls[-1].value:
            num -= (1 << len(ls))
    except TypeError:
        mask = sum(1 << i for i, v in enumerate(ls) if type(v.value) is DontCare)
        num = sum(1 << i for i, v in enumerate(ls) if type(v.value) is not DontCare and v.value)
        return DontCareVal(num, mask)
    return num


def r(p, q):
    (pv, pm) = p
    (qv, _qm) = q
    return (pv & qv), (pv ^ qv) | pm


def calc_val(self, args):
    argvals = (calc_all(a) for a in args)
    v, m = reduce(r, ((self.func(*av), 0) for av in itertools.product(*argvals)))
    val = DontCareVal(v, m)
    return val


def calc_all(a):
    if type(a) is DontCare:
        bits = [b for b in
                itertools.takewhile(lambda x: x <= a.mask, map(lambda x: 1 << x, itertools.count()))
                if b & a.mask]
        aval = (a.val & ~a.mask)
        val = map(lambda x: aval | sum(itertools.compress(bits, x)),
                  itertools.product([0, 1], repeat=len(bits)))
    else:
        val = [a]
    return val


class Signal(object):
    __slots__ = "value", "fanout", "vec"
    def __init__(self):
        self.value = None
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

    def __bool__(self):
        return self.value

    def __iter__(self):
        return iter([self])

    def eq(self, other):
        return Not(Xor(self, other))

    def neq(self, other):
        return Xor(self, other)

    def __hash__(self):
        return id(self)

    def concat(self, *others):
        return Vector(self).concat(*others)

    def simulate(self):
        simulate([self])

    def eval(self):
        args = (a.value for a in self.args)
        try:
            value = self.func(*args)
        except:
            value = calc_val(self, args)

        if self.value == value:
            return []
        else:
            self.value = value
            return self.fanout

    def find_path(self, sigs, oldSigs):
        # print len(sigs)
        if self in sigs:
            self._path = [self]
            return
        allArgs = [s.fanout for s in sigs]
        allArgSet = set()
        allArgSet.update(*allArgs)
        allArgSet.difference_update(oldSigs)
        if not allArgSet:
            self._path = []
            return
        oldSigs.update(allArgSet)
        self.find_path(list(allArgSet), oldSigs)
        if self._path:
            self._path.append(sigs[[n for n, a in enumerate(allArgs) if self._path[-1] in a][0]])

    def checkEqual(self, val):
        if val != self.value:
            print(self, "current result", val, "not equal to", self.value)

    def setArgs(self, args, check=False):
        if check:
            assert not self.args
            value = self.value

        if args:
            for a in args:
                if isinstance(a, Vector):
                    print(a)
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
                    print("whoops!", self, arg, arg.fanout)
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
        res.setArgs(args, True)
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

    def simplify(self, D):
        args = self.simplifyTop(D)
        return self.simplifyBottom(self, args, D)


class FeedbackSignal(Signal):
    def __init__(self, arg):
        Signal.__init__(self)
        self._path = []
        self.setArgs((arg,))

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
        sig.find_path(list(sig.fanout), set())
        simulate(self.setArgs((sig,)))

    def func(self, arg):
        return arg


class Vector(object):
    def __init__(self, bits):
        self.stack = inspect.currentframe().f_back
        self.enum = None
        self.bits = bits
        self._ls = None
        self.check = True

    post_connects = []

    @property
    def ls(self):
        if self._ls is None:
            do_post = Vector.post_connects == []
            # assert isinstance(self, ConstantVector) or self not in Vector.post_connects
            Vector.post_connects.append(self)
            self._ls = list(self.signals())
            assert len(self._ls) == self.bits
            for s in self._ls:
                assert isinstance(s, Signal)  # don't combine vectors into new vector, use concat for that
                if s.vec is None:
                    s.vec = self

            if do_post:
                for v in Vector.post_connects:
                    v.connect_signals()
                Vector.post_connects = []
        return self._ls

    @ls.setter
    def ls(self, value):
        self._ls = value

    def makeEnum(self, enum):
        self.enum = enum
        return self

    def toUint(self):
        return self.current()

    def __repr__(self):
        if self.enum:
            return '{0}({1})'.format(type(self).__name__,
                                     ([k for k in self.enum if int(self.enum[k]) == int(self)] or ['?'])[0])
        else:
            val = self.toUint()
            return '{0}({1}, {2})'.format(type(self).__name__, val, len(self))

    def __len__(self):
        return self.bits

    def __getitem__(self, key):
        if isinstance(key, Vector):
            return Multiplexer(key, self.ls)

        def current_func(start, stop):
            mask = ((1 << (stop - start)) - 1)

            def inner(val):
                return val >> start & mask

            return inner

        def signals():
            if not isinstance(key, slice):
                return [self.ls[key]]
            else:
                return self.ls[key]

        res = CircuitVector(1 if not isinstance(key, slice) else len(list(range(self.bits))[key]),
                            signals=signals,
                            func=current_func(
                                (self.bits + key if key < 0 else key) if not isinstance(key, slice) else
                                0 if key.start is None else
                                self.bits + key.start if key.start < 0 else key.start,
                                ((self.bits + key if key < 0 else key) + 1) if not isinstance(key, slice) else
                                self.bits if key.stop is None else
                                self.bits + key.stop if key.stop < 0 else key.stop),
                            args=[self])
        assert (not isinstance(key, slice) or key.step is None)
        return res

    def __int__(self):
        val = self.toUint()
        if (val << 1) > (1 << self.bits):
            val -= (1 << self.bits)
        return val

    def __iter__(self):
        return (self[n] for n in range(len(self)))

    def __add__(self, other):
        def signals():
            old = KoggeStoneAdder(self, other)[0]
            return old.ls

        if not isinstance(other, Vector):
            other = ConstantVector(other, self.bits)
        mask = (1 << self.bits) - 1
        res = CircuitVector(len_all(self, other), signals=signals,
                            func=lambda a, b: (a + b) & mask, args=[self, other])
        return res

    def add_carry(self, other, c=False):
        def signals():
            old, old_c = KoggeStoneAdder(self, other, c)
            return old.ls

        mask = (1 << self.bits) - 1
        res = CircuitVector(len_all(self, other), signals=signals,
                            func=lambda a, b: (a + b) & mask, args=[self, other])
        return res

    def __sub__(self, other):
        return KoggeStoneAdder(self, ~other, True)[0]

    def __mul__(self, other):
        return Multiplier(self, other)

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
        return KoggeStoneAdder(inv, ConstantVector(0, len(self)), True)[0]

    def __pos__(self):
        return self

    def __abs__(self):
        return If(self[-1], -self, +self)

    def __invert__(self):
        return CircuitVector.op(__invert__, self)

    def __lt__(self, other):
        return int(self) < int(other)

    def __le__(self, other):
        return int(self) <= int(other)

    def __eq__(self, other):
        def signals():
            return [And(*[s.eq(o) for s, o in zip(self.ls, other.ls)])]

        if not isinstance(other, Vector):
            other = ConstantVector(other, self.bits)
        len_all(self, other)
        res = CircuitVector(1, signals=signals,
                            func=lambda a, b: int(a == b), args=[self, other])
        return res

    def __ne__(self, other):
        def signals():
            return [And(*[s.neq(o) for s, o in zip(self.ls, other.ls)])]

        len_all(self, other)
        res = CircuitVector(1, signals=signals,
                            func=lambda a, b: int(a != b), args=[self, other])
        return res

    def __gt__(self, other):
        return int(self) > int(other)

    def __ge__(self, other):
        return int(self) >= int(other)

    def __bool__(self):
        return int(self) != 0

    def __hash__(self):
        return id(self)

    def concat(self, *others):
        def current_func(*vecs):
            sizes = [len(v) for v in vecs]

            def inner(*vals):
                res = 0
                for i, val in enumerate(vals):
                    res |= val << sum(sizes[:i])
                return res

            return inner

        def signals():
            lss = [o.ls for o in others]
            return self.ls + sum(lss, [])

        res = CircuitVector(len(self) + sum(len(o) for o in others), signals=signals, func=current_func(self, *others),
                            args=[self] + list(others))
        return res

    def concatRev(self, *others):
        vectors = [self]
        vectors.extend(others)
        vectors.reverse()
        return Vector.concat(*vectors)
        return

    def extendBy(self, l, LSB=False, signed=False, signal=None):
        if signal is None:
            signal = VFalse
        assert isinstance(signal, Vector)
        if len(signal) != 1:
            raise TypeError('Vector given as signal argument is not of length 1')

        if signed:
            signal = self[-1]
        if LSB:
            return signal.dup(l).concat(self)
        else:
            return self.concat(signal.dup(l))

    def extendTo(self, l, LSB=False, signed=False, signal=None):
        return self.extendBy(l - len(self), LSB, signed, signal)

    def dup(self, n):
        def current_func(num):
            mask = (1 << num) - 1
            def inner(sigv):
                dupbits = mask
                if isinstance(sigv, DontCare):
                    if sigv.mask & 1:
                        return DontCareVal(0, dupbits)
                    else:
                        return 0
                return dupbits if (sigv & 1) else 0

            return inner

        def signals():
            return self.ls[0:1] * n

        if n == 1:
            return self
        return CircuitVector(n, signals=signals, func=current_func(n), args=[self])

    # op is an CircuitOper class that is applied to all signals of self. returns a 1-vector
    def reduce(self, op):
        # obj = op(*self.ls)
        obj = op()

        def signals():
            res = op(*self.ls)
            return res

        def current_func(arg):
            return obj.func(*intToBits(arg, len(self)))

        return CircuitVector(1, signals=signals, func=current_func, args=[self])

    @classmethod
    # apply function pairwise on each signal in the argument vectors
    # fn is a function of multiple arguments that evaluates both on bool arguments
    # as on ints, treating them bitwise.
    def op(cls, fn, *args):
        bits = len_all(*args)
        mask = (1 << bits) - 1

        def signals():
            return [fn(*arg) for arg in zip_all(*(a.ls for a in args))]

        def current_func(*val):
            vec = fn(*val)
            return vec & mask

        return CircuitVector(bits, signals=signals, func=current_func, args=args)

    def checkEqual(self, val):
        if not self.check:
            return
        signals_val = signalsToInt(self.ls, False)
        if val != signals_val:
            print("Current result", val, "not equal to", signals_val)
            assert False
            current_cache.clear()
            print(self.current())
            stack = self.stack
            for _ in range(6):
                print('File "{0}", line {1}, in {2}'.format(stack.f_code.co_filename, stack.f_lineno,
                                                            stack.f_code.co_name))
                stack = stack.f_back
                if not stack:
                    break
            if isinstance(self, FeedbackVector):
                return
            if not self.func:
                return
            try:
                print('operation: File "{0}", line {1}, in {2}'.format(self.func.__code__.co_filename,
                                                                       self.func.__code__.co_firstlineno,
                                                                       self.func.__name__))
            except:
                print('operation: <builtin> {0}'.format(self.func.__name__))
            print("on arguments:", self.args)
            assert False

    def connect_signals(self):
        pass


def makeVector(ls):
    res = Vector(len(ls))
    res._ls = ls
    return res


class CircuitVector(Vector):
    def __init__(self, bits, signals, func=None, args=None, enum=None):
        Vector.__init__(self, bits)
        if args is None:
            args = []
        # assert func != None
        for a in args:
            assert isinstance(a, Vector), (type(a), a)
        self.signals = signals
        self.func = func
        self.args = args
        self.enum = enum

    def current(self):
        if self in current_cache:
            return current_cache[self]

        args = [a.current() for a in self.args]
        try:
            val = self.func(*args)
        except TypeError as e:
            val = calc_val(self, args)

        current_cache[self] = val
        self.checkEqual(val)
        return val


class ValueVector(Vector):
    def __init__(self, value, bits):
        Vector.__init__(self, bits)
        try:
            self.value = int(value) & (1 << bits) - 1
        except TypeError:
            self.value = value

    def current(self):
        return self.value


class ConstantVector(ValueVector):
    def __init__(self, value, bits=16, enum=None):
        ValueVector.__init__(self, value, bits)
        if enum:
            self.makeEnum(enum)

    def signals(self):
        ls = [ConstantSignal(b) for b in intToBits(int(self.value), self.bits)]
        return ls

    def __hash__(self):
        return int(self)

    def __eq__(self, other):
        return isinstance(other, Vector) and int(self) == int(other)


VTrue = ConstantVector(True, 1)
VFalse = ConstantVector(False, 1)


class TestVector(ValueVector):
    def __init__(self, value, bits=16):
        ValueVector.__init__(self, value, bits)
        ls = self.signals()

    def signals(self):
        ls = [TestSignal(b) for b in intToBits(int(self.value), self.bits)]
        return ls

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            signals = self.ls[key]
        if isinstance(value, int):
            bits = intToBits(value, len(signals))
        elif isinstance(value, DontCare):
            bits = DontCareToBits(value, len(signals))
        else:
            bits = value
        for b, s in zip(bits, signals):
            s.value = b.value if isinstance(b, Signal) else b
        try:
            assert key == slice(None, None, None)
            self.value = int(value) & (1 << len(signals)) - 1
        except TypeError:
            self.value = value
        current_cache.clear()
        simulate(signals)


class Clock(TestVector):
    def __init__(self, initval=False):
        TestVector.__init__(self, initval, 1)
        self.registers = []

    def cycle(self, n=1, CPU_state=None):
        for _ in range(n):

            newvals = [(r, r.next.current()) for e, r in enumerate(self.registers)]
            self[:] = 1
            self[:] = 0
            for r, v in newvals:
                r.value = v
                assert r.current() == v

            if CPU_state is not None:
                print("===============================")
                for k in CPU_state:
                    print(k, ":", CPU_state[k])


class FeedbackVector(Vector):
    def __init__(self, value=0, bits=1, name=None):
        self.arg = None
        self.name = name if name is not None else str(type(self).__name__) + str(id(self))

        if isinstance(value, Vector):
            bits = value.bits
        Vector.__init__(self, bits)
        if isinstance(value, Vector):
            if value.enum:
                self.makeEnum(value.enum)
            value = value.current()
        self.value = value

    def signals(self):
        self._ls = [FeedbackSignal(b) for b in intToBits(int(self.value), self.bits)]
        return self._ls

    def connect(self, vec):
        self.arg = vec
        if self._ls is not None:
            self.connect_signals()

    def connect_signals(self):
        if self.arg is not None:
            for my_sig, other_sig in zip_all(self._ls, self.arg.ls):
                my_sig.connect(other_sig)

    def current(self):
        if self in current_cache:
            if current_cache[self] is None:
                return self.value
            else:
                return current_cache[self]
        current_cache[self] = None

        val = self.arg.current()

        current_cache[self] = val
        self.checkEqual(val)
        return val


class DontCare(object):
    def __init__(self, val=False, mask=1):
        self.val = val
        self.mask = mask

    def __bool__(self):
        assert self.mask == 1
        return DontCareBool

    def __eq__(self, other):
        return type(self) == type(other) and self.val == other.val and self.mask == other.mask

    def __ne__(self, other):
        return not self.__eq__(other)

    def __and__(self, other):
        if isinstance(other, DontCare):
            new_val = self.val & other.val
            new_mask = (self.val | self.mask) & (other.val | other.mask) & ~new_val
        else:
            new_val = self.val & other
            new_mask = self.mask & other
        if new_mask:
            return DontCareVal(new_val, new_mask)
        else:
            return new_val

    def __rand__(self, other):
        return self.__and__(other)

    def __or__(self, other):
        if isinstance(other, DontCare):
            new_val = self.val | other.val
            new_mask = (self.mask | other.mask) & ~(self.val | other.val)
        else:
            new_val = self.val | other
            new_mask = self.mask & ~other
        if new_mask:
            return DontCareVal(new_val, new_mask)
        else:
            return new_val

    def __ror__(self, other):
        return self.__or__(other)

    def __xor__(self, other):
        if isinstance(other, DontCare):
            new_mask = self.mask | other.mask
            new_val = (self.val ^ other.val) & ~new_mask
        else:
            new_mask = self.mask
            new_val = (self.val ^ other) & ~new_mask
        return DontCareVal(new_val, new_mask)

    def __rxor__(self, other):
        return self.__xor__(other)

    def __invert__(self):
        return DontCareVal(~(self.val | self.mask), self.mask)

    def __lshift__(self, other):
        return DontCareVal(self.val << other, self.mask << other)

    def __rshift__(self, other):
        return DontCareVal(self.val >> other, self.mask >> other)

    def __repr__(self):
        return "DontCare(" + str(self.val) + ", " + str(self.mask) + ")"

    def __format__(self, format_spec):
        return ("DontCare(" + str(self.val) + ", " + str(self.mask) + ")").__format__(format_spec)


DontCareBool = DontCare(False, 1)


def DontCareVal(v, m):
    val = v if m == 0 else \
        DontCareBool if m == 1 and type(v) is bool else \
            DontCare(v, m)
    return val


class DontCareSignal(ConstantSignal):
    def __init__(self):
        ConstantSignal.__init__(self)
        self.value = DontCareBool

    def __repr__(self):
        return '{0}()'.format(type(self).__name__)


DontCareSig = DontCareSignal()


class DontCareVector(ValueVector):
    def __init__(self, bits=16):
        mask = (1 << bits) - 1
        value = DontCare(False, mask)
        ValueVector.__init__(self, value, bits)

    def signals(self):
        ls = [DontCareSig] * self.bits
        return ls

    def __repr__(self):
        return "DontCareVector({0})".format(self.bits)


class CircuitOper(Signal):
    def __init__(self, *args):
        Signal.__init__(self)
        self.setArgs(args)

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
        elif not cvals:
            res = self
        else:
            if not vs:
                res = STrue if self.func(*cvals) else SFalse
                args = vs
            else:
                res, args = self.partial(vs, cvals, D)
        assert res.value == self.value
        return self.simplifyBottom(res, args, D)


class And(CircuitOper):
    def func(self, *a):
        return reduce(and_, a, True)

    @staticmethod
    def constPartial(v, cf):
        return v if cf else SFalse


class Or(CircuitOper):
    def func(self, *a):
        return reduce(or_, a, False)

    @staticmethod
    def constPartial(v, cf):
        return STrue if cf else v


class Xor(CircuitOper):
    def func(self, *a):
        return reduce(xor, a)

    @staticmethod
    def constPartial(v, cf):
        return Not(v) if cf else v


class Not(CircuitOper):
    def func(self, a):
        if isinstance(a, DontCare):
            return DontCareBool
        else:
            return not a


class Nor(CircuitOper):
    def func(self, *a):
        or_val = reduce(or_, a, False)
        if isinstance(or_val, DontCare):
            return DontCareBool
        else:
            return not or_val


class Eq(CircuitOper):
    def func(self, *a):
        return reduce(eq, a)


class Ne(CircuitOper):
    def func(self, *a):
        return reduce(ne, a)


class Enum(object):
    def __init__(self, *args):
        if len(args) == 1:
            try:
                kv = [(k, v) for v, k in args[0].items()]
            except:
                kv = list(enumerate(args))
        else:
            kv = list(enumerate(args))
        bits = 0
        maxarg = max(k for k, _ in kv)
        while maxarg >= (1 << bits):
            bits += 1

        self._dict = {k: ConstantVector(v, bits, self) for v, k in kv}

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
    for k in list(cases.keys()):
        if not (isinstance(k, Vector) or isinstance(k, Signal) or isinstance(k, str)):
            v = cases[k]
            for kk in k:
                cases[kk] = v
            del cases[k]
    if isinstance(state, Vector) and state.enum:
        cases = {state.enum[k] if isinstance(k, str) else k: cases[k] for k in cases}
        for k in cases:
            if k.enum != state.enum:
                raise ValueError("%s and %s are not of same enum type")
    length = len_all(state, *list(cases.keys()))
    if default is None:
        default = getDefault(*list(cases.values()))
    else:
        getDefault(default, *list(cases.values()))
    intcases = {k.toUint(): cases[k] for k in cases}
    alts = [intcases.get(i, default) for i in range(2 ** length)]
    return Multiplexer(state, alts)


def TrueInCase(state, cases):
    if isinstance(cases, Vector):
        cases = [cases]
    new_cases = {}
    for k in cases:
        try:
            new_cases[k] = cases[k]
        except TypeError as e:
            new_cases[k] = VTrue
    first = list(new_cases.keys())[0]
    if isinstance(first, Vector) and first.enum is not None:
        allcases = set(first.enum[k] for k in first.enum)
        notcases = allcases - set(new_cases.keys())
        for k in notcases:
            new_cases[k] = VFalse
        return Case(state, new_cases)

    return Case(state, new_cases, default=VFalse)


def HalfAdder(a, b):
    s = a ^ b
    c = a & b
    return s, c


def FullAdder(a, b, c_in):
    s1, c1 = HalfAdder(a, b)
    s2, c2 = HalfAdder(s1, c_in)
    c_out = c1 | c2
    return s2, c_out


def RippleCarryAdder(als, bls, c=False):
    if isinstance(c, bool):
        c = ConstantVector(c, 1)
    assert isinstance(c, Vector)
    assert isinstance(als, Vector)
    assert isinstance(bls, Vector)
    assert len(als) == len(bls)
    sls = ConstantVector(0, 0)
    for a, b in zip_all(als, bls):
        s, c = FullAdder(a, b, c)
        sls = Vector.concat(sls, s)
    return sls, c


def KoggeStoneAdder(als, bls, c=False):
    if isinstance(c, bool):
        c = ConstantVector(c, 1)
    assert isinstance(c, Vector)
    assert isinstance(als, Vector)
    assert isinstance(bls, Vector)
    assert len(als) == len(bls)
    assert isinstance(c, Vector) and len(c) == 1
    prop = {0: als ^ bls}
    gen = {0: als & bls}

    step = 1
    while step < len(als):
        p = prop[step // 2]
        g = gen[step // 2]

        p_prev = p[:-step].extendTo(len(p), LSB=True)
        g_prev = g[:-step].extendTo(len(g), LSB=True, signal=c)
        prop[step] = p & p_prev
        gen[step] = (p & g_prev) | g
        step *= 2

    cls = c.concat(gen[step / 2])
    sls = Vector.op(lambda a, b, c: a ^ b ^ c, als, bls, cls[:-1])
    return sls, cls[-1]


def DecimalAdder(als, bls, c):
    if isinstance(c, bool):
        c = ConstantVector(c, 1)
    assert isinstance(c, Vector)
    assert isinstance(als, Vector)
    assert isinstance(bls, Vector)
    assert len(als) == len(bls)
    nibbles = [(als[i:i + 4].extendBy(1), bls[i:i + 4].extendBy(1)) for i in
               range(0, len(als), 4)]
    dst_out = []
    for src_n, dst_n in nibbles:
        sum_n, c_out = KoggeStoneAdder(src_n, dst_n, c)
        adjusted, adc_out = KoggeStoneAdder(sum_n, ConstantVector(-10, 5))

        c = ~adjusted[-1]
        dst_out.append(If(c, adjusted[:-1], sum_n[:-1]))

    return Vector.concat(*dst_out), c


def Multiplier(als, bls, signed=True):
    if signed:
        als_sign = als[-1]
        bls_sign = bls[-1]
        als = If(als_sign, -als, als)
        bls = If(bls_sign, -bls, bls)

    sls, c = ConstantVector(0, len(bls) - 1), VFalse
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
    mask = (1 << len(data)) - 1
    initvalues = [i & mask for i in (init + [0] * ((1 << len(addr)) - len(init)))]


    def signals():
        wordlines = Decoder(addr)  # [c & r for c in collines for r in rowlines]
        doWrite = isWrite & ~clk
        isWriteVec = doWrite.dup(len(data))
        sets = Vector.op(__and__, isWriteVec, data)
        resets = Vector.op(__and__, isWriteVec, ~data)
        q_outs = [[SRLatch(access & s, access & r, b)[0] & access
                   for (s, r, b) in zip_all(sets.ls, resets.ls, list(intToBits(val, len(data))))]
                  for access, val in zip_all(wordlines.ls, initvalues)]
        return [Or(*l) for l in zip(*q_outs)]

    res = CircuitVector(len(data), signals=signals, func=current_func(initvalues), args=[addr, data, isWrite])
    return res


def ROM(_, addr, data, __, init):
    def current_func(values):
        def inner(mem_addr, _data):
            return values[mem_addr]

        return inner

    if not init:
        init = []
    mask = (1 << len(data)) - 1
    initvalues = [i & mask for i in (init + [0] * ((1 << len(addr)) - len(init)))]

    wordlines = Decoder(addr)

    def signals():
        q_outs = [(ConstantVector(val) & access).ls for access, val in zip_all(wordlines, initvalues)]
        return [Or(*l) for l in zip(*q_outs)]

    res = CircuitVector(len(data), signals=signals, func=current_func(initvalues), args=[addr, data])
    return res


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
            return (makeVector(alts) & enables).reduce(Or)
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

            def signals():
                ls = [combiner([a.ls[i] for a in alts]) for i in range(ln)]
                for e in ls:
                    assert len(e) == 1
                return [e.ls[0] for e in ls]

            return CircuitVector(ln, signals=signals, func=current_func(alts),
                                 args=[sel])
        else:
            typ = type(alt0)
            ln = len(alt0)
            for alt in alts[1:]:
                if type(alt) is not typ:
                    raise TypeError()
                if len(alt) is not ln:
                    raise ValueError()
            return typ(combiner(a[i] for a in alts) for i in range(ln))

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
        return makeVector([])
    return makeVector(outs[0])


class Register(FeedbackVector):
    def __init__(self, clk, value=0, bits=1, name=None):
        def inner():
            return self.current()

        FeedbackVector.__init__(self, value, bits, name)
        self.clk = clk
        self.next = None
        clk.registers.append(self)
        self.func = inner
        self.args = []

    def connect(self, next):
        self.next = next

        if self._ls is not None:
            self.connect_signals()
        return self

    def connect_signals(self):
        if self.next is not None:
            self.arg = FlipFlops(self.next.ls, self.clk.ls[0], self._ls)
            FeedbackVector.connect_signals(self)
            self.checkEqual(self.value)

    def ccase(self, state, cases, default=None):
        if default is None: default = self
        self.connect(Case(state, cases, default))

    def current(self):
        val = self.value

        self.checkEqual(val)
        return val

    def __repr__(self):
        return "{}: {}".format(self.name, FeedbackVector.__repr__(self))
