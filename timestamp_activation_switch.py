from poly_interpolation import *

"""
    description
time-interval probability distribution

functions are of two types:
- polynomial
- dict : k -> v

    arguments
a := (2-col matrix)::(polynomial points) | (dict,mode = 'literal'|'round')

"""
class TimeIntervalPD:

    def __init__(self,a):
        self.a = a
        self.f = None
        self.make_func()

    def make_func(self):
        if not (type(self.a) is np.ndarray):
            self.f = self.make_dfunc(self.a)
            return
        self.f = self.make_pfunc(self.a)

    def make_dfunc(self, d):
        assert type(d[0]) is dict, "invalid dict."
        s = None
        if d[1] == 'literal':
            def f(x):
                if x not in d[0]: return None
                return d[0][x]
            print("making literal")
            s = f
        elif d[1] == 'round':
            ks = np.array(list(d.keys()))
            def f(x):
                if x not in ks:
                    i = np.argmin(np.abs(ks - x))
                    return d[ks[i]]
                return d[x]
            s = f
        else:
            raise ValueError("invalid args. for dfunc")
        return s

    def make_pfunc(self, twoColumnMatrix):
        assert type(twoColumnMatrix) is np.ndarray, "invalid matrix"
        assert twoColumnMatrix.shape[1] == 2, "invalid matrix shape"
        lps = LagrangePolySolver(points, prefetch = True)
        return lps.output_by_lagrange_basis

    def __getitem__(self,t):
        return self.f(t)

"""
Switch is active for the duration t = 0...max(l).

At each timestamp, instance will draw a probability value `pv`
from TimeIntervalPD, then will call action function `bFunc` on `pv`
to output a bool or value `o`.

Additionally, if tFunc is not None, then at each timestamp, if
tFunc(`o`), then terminate.
"""
class TimestampActivationSwitch:

    def __init__(self,t, l, mFunc, bFunc, tFunc, incFunc = lambda x: x + 1):
        self.t = t
        self.l = l
        self.bFunc = bFunc
        self.iFunc = incFunc
        self.mFunc = mFunc
        self.tFunc = tFunc
        self.terminated = False

    def passed_limit(self):
        return self.t > self.l

    def __next__(self):
        if self.passed_limit(): return None
        if self.terminated: return None

        # get pr. value
        pr = self.mFunc[self.t]

        # get pr-based value
        r = self.bFunc(pr)

        if type(self.tFunc) != type(None):
            self.terminated = self.tFunc(r)

        # update time
        t0 = self.t
        self.t = self.iFunc(self.t)
        return t0, pr,r,self.terminated
