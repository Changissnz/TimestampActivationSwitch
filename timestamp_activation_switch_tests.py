from timestamp_activation_switch import *

################### TODO: run tests here.

# bFunc := float
sample_bfunc_float = lambda pr: int(ceil(pr * 1234))

def sample_bfunc_vector(pr):
    return "DO IT"

sample_tfunc = lambda x: x % 7 < 3

def sample_tfunc_activation_range(ar):
    assert ar[0] <= ar[1], "invalid range, must be ordered"
    def in_range(x):
        return x >= ar[0] and x <= ar[1]
    return in_range

def sample_tfunc_true():
    return lambda t: t == True

#    def __init__(self,t, l, mFunc, bFunc, tFunc, incFunc = lambda x: x + 1):

"""
case 1: uses
"""
def case_1():

    t = 0.0
    l = 50


    # make a dictionary 
    mFunc =


# make a sample LagrangePolySolver w/ the points
'''
t  pr
0   0.5
5   0.25
10  0.5
15  0.25
20  0.75
'''
