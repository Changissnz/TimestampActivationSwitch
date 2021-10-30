from timestamp_activation_switch import *

################### TODO: run tests here.

# bFunc := float
sample_bfunc_float = lambda pr: int(math.ceil(pr * 1234)) if type(pr) != type(None) else None

def sample_bfunc_vector(pr):
    return "DO IT"

sample_tfunc = lambda x: x % 7 < 3 if type(x) != type(None) else None

def sample_tfunc_activation_range(ar):
    assert ar[0] <= ar[1], "invalid range, must be ordered"
    def in_range(x):
        return x >= ar[0] and x <= ar[1]
    return in_range

def sample_tfunc_true():
    return lambda t: t == True

"""
case 1: uses
"""
def TimestampActivationSwitch__case_1():

    t = 0.0
    l = 50

    # make a dictionary
    mFunc = {}
    for i in range(10):
        mFunc[float(i)] = 0.2
    for i in range(20):
        mFunc[float(i + 10 * 1)] = 0.4
    for i in range(10):
        mFunc[float(i + 10 * 3)] = 0.9
    for i in range(10):
        mFunc[float(i + 10 * 4)] = 0.1

    print("Q? ",10 in mFunc)
    print("Q? ",mFunc[20])


    mf = (mFunc,'literal')
    mf = TimeIntervalPD(mf)

    bf = sample_bfunc_float
    tFunc = sample_tfunc
    #tFunc = None
    iFunc = lambda x: x + 1
    return TimestampActivationSwitch(t,l,mf,bf,tFunc,iFunc)

def test__case_1():
    tas = TimestampActivationSwitch__case_1()
    i = 0
    while i < 50 and not tas.terminated:
        print(next(tas))
        i += 1
    assert i == 1, "incorrect for term. mode"

    print("NO TERM")
    tas = TimestampActivationSwitch__case_1()
    tas.tFunc = None
    for i in range(50):
        print(next(tas))

# make a sample LagrangePolySolver w/ the points
'''
t  pr
0   0.5
5   0.25
10  0.5
15  0.25
20  0.75
'''
