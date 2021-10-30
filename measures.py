from matrix_methods import *

########## START: stat measures 

def standard_variance_on_sequence(npSeq):
    assert len(npSeq.shape) == 1, "invalid dim., want 1" 
    return npSeq.var() 

# TODO: in testing
def opposing_deviation_on_sequence(npSeq):
    assert len(npSeq.shape) == 1, "invalid dim., want 1" 
    m = npSeq.mean()
    diff = np.array([x - m for x in npSeq])
    return sum(diff) 

# TODO: opposing variance w/in e-ball 

########## END: stat measures 

########## START: point measures 

zero_div = lambda num, denum, default: default if denum == 0 else num / denum 

"""
description: 
- standard point-distance formula 

arguments: 
- p1, p2: 2-d iterables, points
"""    
point_distance = lambda p1, p2: math.sqrt((p1[0] - p2[0]) **2 + (p1[1] - p2[1]) ** 2) 

"""
description: 
- summation of total distances from one point to iterable of points

arguments: 
- p1: 2-d iterable 
- ps: iterable, 2-d points 
"""
point_distance_one_to_rest = lambda p1, ps: sum(point_distance(p1, ps_) for ps_ in ps)

"""
description: 
- determines which points in `ps` are <= `threshold` distance from `p1` 
"""
point_distance_below_threshold_one_to_rest = lambda p1, threshold, ps: np.array([ps_ for ps_ in ps if point_distance(p1, ps_) <= threshold]) 

# TODO: check for 0 case
point_slope = lambda x, x2: (x2[1] - x[1]) / (x2[0] - x[0]) if  abs(x2[0] - x[0]) >= 10 ** -5 else 0

midpoint = lambda x, x2: np.array([(x[0] + x2[0]) / 2, (x[1] + x2[1]) / 2])

def angle_between_two_points(startX, endX):    
    if equal_iterables(startX, endX): 
        return 0 

    l = point_distance(startX, endX) 
    opposite = endX[1] - startX[1]

    q = opposite / float(l)
    return math.degrees(np.arcsin(q))

"""
"""
def angle_between_two_points_clockwise(startX, endX): 
    angle = angle_between_two_points(startX, endX)
    if angle == 0: return 0

    # case: forward
    if startX[0] < endX[0]:
        if angle > 0: 
            return angle 
        else: 
            return angle % 360 
    else: 
        return 180 - angle
    return angle 

def angle_between_two_angles(angle, ref, end): 
    assert angle >= 0 and angle < 360, "invalid angle"
    assert ref >= 0 and ref < 360, "invalid angle"
    assert end >= 0 and end < 360, "invalid angle"

    # case 0 
    if ref > end: 
        stat = angle >= ref and angle < 360 # ref to 360
        stat2 = angle >= 0 and angle <= end # 0 to end 
        if stat or stat2: return True
        return False 

    # case 1 
    else: 
        return True if angle >= ref and angle <= end else False

def positive_complement_of_angle(angle): 
    assert angle >= 0 and angle < 360, "invalid angle"
    return 360 - angle

# TODO: untested
def directional_degree_to_clockwise(startX, endX, degree): 

    if not (abs(degree) <= 90.0 and abs(degree) >= 0): 
        assert abs(degree) >= 0 and abs(degree) <= 360, "invalid degree" 
        return degree  

    if startX == endX: 
        if degree - 90 < 10 ** -5: 
            return degree 

        elif degree + 90 < 10 ** -5: 
            return 270 
        
        raise ValueError("wrong dir. deg.")

    if startX < endX and degree > 0: 
        return degree

    if startX < endX and degree < 0: 
        return degree % 360

    if startX > endX and degree >= 0: 
        return 180 - degree 

    if startX > endX and degree < 0: 
        return 180 + degree 

    return degree

# TODO: 
def clockwise_degree_to_directional(degree): 
    assert abs(degree) >= 0 and abs(degree) < 360, "dsfasdfjas;ldjf asl;df jasdl;f alskd;fj lasd;lj kfasdlkj; faklj sdflk as"
    
    # case: quad 0 and 1
    if degree >= 0 and degree <= 90: 
        return degree 

    if degree > 90 and degree <= 180: 
        return 180 - degree

    if degree > 180 and degree <= 270: 
        return 180 - degree 
    
    if degree > 270 and degree <= 360: 
        return 360 - degree

"""
description: 
- calculates the slope b/t every contiguous point pairs. 

arguments: 
- points: 2-d array
"""
def contiguous_point_slopes(points):
    slopes = [] 
    for i in range(0, points.shape[0] - 1): 
        p1, p2 = points[i], points[i + 1]
        slopes.append(point_slope(p1, p2))
    return np.array(slopes)

# TODO: refactor for n-degree?? 
def contiguous_point_acceleration(points): 
    s1 = contiguous_point_slopes(points) 
    return contiguous_point_slopes(s1)

########## END: point measures 

########## START: array measures 


def partition_average_of_array(a2d, targetAxis, ph:int):
    
    if len(a2d.shape) != 2: 
        raise ValueError("array must be 2-d")
    
    if targetAxis != 0 and targetAxis != 1: 
        raise ValueError("target axis is invalid, 0 OR 1") 
    
    average = []
    index = 0 
    otherIndex = 0 if targetAxis == 0 else 1 
    
    while index < a2d.shape[otherIndex]: 
        nextIndex = index + ph
        
        if otherIndex == 0: 
            mv = a2d[index:nextIndex].mean(axis = targetAxis) 
        else: 
            mv = a2d[:,index:nextIndex].mean(axis = targetAxis) 
        
        average.append(mv) 
        index = nextIndex
    return np.array(average)

# CAUTION: no arg. check 
def add_noise_to_point_sequence(points, maxNoisePerAxis, restrictedAxis, signRestriction = 0):
    return np.array([noise_to_point(p, maxNoisePerAxis, restrictedAxis, signRestriction) for p in points])

# CAUTION: no arg. check
def noise_to_point(point, maxNoisePerAxis, restrictedAxis = None, signRestriction = 0): 
    assert restrictedAxis in [None, "0","1"], "invalid restricted axis {}".format(restrictedAxis)
    assert maxNoisePerAxis >= 0, "invalid max noise, must be non-negative"

    deltaX = random.random() * maxNoisePerAxis
    deltaY = random.random() * maxNoisePerAxis

    if signRestriction == 1:
        pass 
    elif signRestriction == -1: 
        deltaX, deltaY = -deltaX, -deltaY
    
    else: 
        if random.random() > 0.5: 
            deltaX *= -1

        if random.random() > 0.5: 
            deltaY *= -1
            
    if restrictedAxis == "0": 
        deltaX = 0 
    elif restrictedAxis == "1": 
        deltaY = 0 

    return (point[0] + deltaX, point[1] + deltaY)

# TODO: make noise to point 

# TODO: TODO: 
def visualize_adding_different_noise_amplitude_to_line(line, folderPath, numImages, numAmplitudes = 10, maxAmplitude = 10, signRestriction = 0):
    assert numImages >= numAmplitudes and  numImages // numAmplitudes == numImages / numAmplitudes, "invalid # of images and amplitudes!"
    
    def plot_and_save(filePath):
        return -1

    numPicsPer = numImages // numAmplitudes
    amplitudeIncrement = maxAmplitude / numAmplitudes

    line.form_point_sequence("exact") 

    for i in range(numAmplitudes):
        q = amplitudeIncrement * (i + 1)
        for j in range(numPicsPer): 
            return -1 
        return -1     
    return -1 

########## END: point measures

########## START: polynomial density measures 

from collections import Counter

# TODO: draft
def polynomial_density_measure_1(points):  
    slopes = list(contiguous_point_slopes(points)) 

    l = points.shape[0] - 1
    q = Counter(slopes)
    q = normalize_dict_values_by_function(q, lambda x: zero_div(x,l,1)) 
    # TODO: add noise modification here
    vq = [v for v in q.values()]
    ###
    return len(q) / l

def normalize_dict_values_by_function(d, f): 
    for k,v in d.items(): 
        d[k] = f(v)
    return d

def accumulate_function_to_dict_values(): 

    return -1


########## END: polynomial density measures


########## START: formulas for equations and geometric shapes

"""
description: 
- 

return: 
- (sol. set)::list, (sol. exist)::bool
"""
def quadratic_formula_solve_for_reals(a,b,c):
    fac = 4 * a * c
    if b ** 2 < fac:
        return [], False
    q = math.sqrt(b ** 2 - fac)
    x1,x2 = (-b + q) / (2 * a), (-b - q) / (2 * a)
    return [x1,x2], True 

# TODO: not tested. 
#       easier approach is distance(p, centerPoint) < radius  
def is_point_in_circle(p, centerPoint, radius):
    
    if not (p[0] >= centerPoint[0] - radius and p[0] <= centerPoint[0] + radius): 
        return False
    
    a = 1
    b = -2 * centerPoint[1]
    c = p[0] ** 2 + centerPoint[0] ** 2 - 2 * centerPoint[0] * p[0] - radius ** 2 + centerPoint[1] ** 2  

    soln, stat = quadratic_formula_solve_for_reals(a,b,c)
    if not stat: return stat 

    if p[1] >= min(soln) and p[1] <= max(soln): return True 
    return False 

def is_line_in_circle(line, circleCenter, circleRadius): 

    # endpoint [0] 
    if is_point_in_circle(line.endpoints[0], circleCenter, circleRadius): 
        return True 

    # endpoint [1] 
    if is_point_in_circle(line.endpoints[1], circleCenter, circleRadius): 
        return True 

    return False 