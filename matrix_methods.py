import numpy as np
import math 
import random 

def equal_iterables(i1, i2, roundPlaces = 5): 

    if len(i1) != len(i2): return False 
    if np.all(np.equal(np.round(i1, 5), np.round(i2, 5)) == True): return True 
    return False

def indices_of_vector_in_matrix(a, v): 
    assert is_2dmatrix(a) and is_vector(v), "invalid criteria"
    a = np.round(a,5) 
    v = np.round(v,5)
    indices = []
    for (i, x) in enumerate(a): 
        if np.all(np.equal(x, v) == True): indices.append(i) 
    return np.array(indices) 

def is_2dmatrix(m):
    if type(m) is not np.ndarray: return False 
    if len(m.shape) != 2: return False 
    return True 

def is_vector(m): 
    if type(m) is not np.ndarray: return False 
    if len(m.shape) != 1: return False 
    return True

def is_valid_point(point):
    assert not type(point) is np.ndarray, "point cannot be np.ndarray"
    if len(point) != 2: return False
    if not type(point[0]) in [int, float, np.int64, np.float64]: return False# or type(point[0]) is float): return False 
    if not type(point[1]) in [int, float, np.int64, np.float64]: return False
    return True 

def np_array_to_string(a): 
    assert type(a) is np.ndarray, "not np.ndarray"
    s = str(a).replace("\n", "\t")  
    return s 

###
"""
def flip_subvector_of_binary_vector(v, start, end, binaryClassification = [1,-1]): 
    assert is_vector(v), "{} is not vector".format(v)
    assert type(binaryClassification) is list and len(binaryClassification) == 2, "invalid binary class."
    # TODO: does not perform element-wise check of vector for binary class.
    assert type(start) is int and (end == None or type(end) is int)
    if end != None: assert end > start and 
    s = v[start:end] if end != None else v[start:]
"""
### 

# TODO: expand on criteria
# TODO: unused
def frequency_count_over_2dmatrix(a, floatValue, rowOrColumn = None): 
    r,c = np.where(a == floatValue)
    if rowOrColumn == "r": return r
    elif rowOrColumn == "c": return c
    return r,c

# TODO: test
def range_intersection(range1,range2): 
    assert type(range1) is tuple and type(range2) is tuple, "invalid arg. tuple"
    assert len(range1) == 2 and len(range2) == 2, "invalid tuple size" 
    assert range1[0] <= range1[1] and range2[0] <= range2[1], "invalid ranges, [min,max]"

    # check range 1
    if range1[0] >= range2[0] and range1[0] <= range2[1]: return True
    if range1[1] >= range2[0] and range1[1] <= range2[1]: return True

    # check range 2
    if range2[0] >= range1[0] and range2[0] <= range1[1]: return True
    if range2[1] >= range1[0] and range2[1] <= range1[1]: return True

    return False

# TODO: test 
def complement_of_range_in_range(range1, range2):
    assert range1[0] <= range1[1], "invalid range 1"
    if range2 == None: return range1 

    assert range2[0] <= range2[1], "invalid range 2"    
    assert range2[0] >= range1[0] and range2[1] <= range1[1], "range 2 not in range 1" 

    filter_null_range = lambda r: False if abs(r[0] - r[1])\
                                <= 10 ** -5 else True 

    q1 = (range1[0], range2[0]) 
    q2 = (range2[1], range1[1]) 
    output = [] 
    
    if filter_null_range(q1): output.append(q1) 
    if filter_null_range(q2): output.append(q2)  

    return output 



################################################ START: area measures 
"""
Area is 2 x 2 matrix in which [0] represents bottom left, [1] represents upper right. 
"""

def is_valid_area(area): 
    return False if not is_2dmatrix(area) else True
    if area.shape[0] != 2 or area.shape[1] != 2: return False 
    if area[0,0] > area[1,0]: return False
    if area[0,1] > area[1,1]: return False
    return True

def dim_of_area(area):
    assert is_valid_area(area), "invalid area"
    return area[1,0] - area[0,0], area[1,1] - area[0,1]

def value_of_area(area): 
    dim = dim_of_area(area) 
    return dim[0] * dim[1]

# TODO: requires Riemann sums; use MP. 
def value_of_area_given_disqualifying(area, disqualifiedAreas): 
    return -1 

"""
is a1 in a2
"""
def area_in_area(a1, a2): 
    assert is_valid_area(a1) and is_valid_area(a2), "input are not areas"
    if not (a1[0,0] >= a2[0,0] and a1[1,0] <= a2[1,0]): return False 
    if not (a1[0,1] >= a2[0,1] and a1[1,1] <= a2[1,1]): return False 
    return True 

def area_intersects(a1, a2):
    assert is_valid_area(a1) and is_valid_area(a2), "input are not areas"

    # left bottom 
        # intersects 
    # bottom left 
    if a1[0,0] >= a2[0,0] and a1[0,0] <= a2[1,0]\
        and a1[0,1] >= a2[0,1] and a1[0,1] <= a2[1,1]: 
        return True

    # upper left 
    if a1[0,0] >= a2[0,0] and a1[0,0] <= a2[1,0]\
        and a1[1,1] >= a2[0,1] and a1[1,1] <= a2[1,1]: 
        return True

    # bottom right 
    if a1[1,0] >= a2[0,0] and a1[1,0] <= a2[1,0]\
        and a1[0,1] >= a2[0,1] and a1[0,1] <= a2[1,1]: 
        return True

    # upper right 
    if a1[1,0] >= a2[0,0] and a1[1,0] <= a2[1,0]\
        and a1[1,1] >= a2[0,1] and a1[1,1] <= a2[1,1]: 
        return True
    return False

# TODO: necessary? 
def is_area_qualified(unqualified, area):
    for x in unqualified: 
        if area_intersects(x, area): return False 
    return True

# TODO: test 
def point_in_area(point, area):
    assert is_valid_point(point) 
    if not (point[0] >= area[0,0] and point[0] <= area[1,0]): 
        return False 
    if not (point[1] >= area[0,1] and point[1] <= area[1,1]):
        return False 
    return True

def is_point_qualified(unqualifiedAreas, point): 

    for x in unqualifiedAreas: 
        if point_in_area(point, x): return False 
    return True

def quadrant_of_corner_point_in_area(cp, area):
    corners = area_to_corners(area)
    indices = indices_of_vector_in_matrix(corners, cp)
    if len(indices) != 1: raise ValueError("quadrant for corner point {} could not be obtained for\n{}".format(cp, corners))
    return indices[0]

############################################ END: area 

############################################ START: qualifying area search

"""
incrementInfo: dict, keys are [str::"axis" = x|y, float::"increment"]
""" 
def extreme_xy_from_point(unqualifiedAreas, area, point, incrementInfo): 
    # set up increment and qualifying methods for point increment
    if incrementInfo["axis"] == "x":
        if incrementInfo["increment"] >= 0: qf = lambda p: True if p[0] <= area[1,0] else False
        else: qf = lambda p: True if p[0] >= area[0,0] else False
        ifunc = lambda p: (p[0] + incrementInfo["increment"], p[1]) 
    elif incrementInfo["axis"] == "y": 
        if incrementInfo["increment"] >= 0: qf = lambda p: True if p[1] <= area[1,1] else False
        else: qf = lambda p: True if p[1] >= area[1,0] else False
        ifunc = lambda p: (p[0], incrementInfo["increment"] + p[1]) 
    else: 
        raise ValueError("invalid axis") 

    # increment
    prev = None
    pp = point 
    while qf(pp):
        if not is_point_qualified(unqualifiedAreas, pp): 
            break
        prev = pp
        pp = ifunc(pp)

    return prev 

# TODO: implement this.
def greatest_qualifying_area_in_vector_direction(vector, unqualifiedAreas, area, incrementDirections, areaWanted):
    return -1 

# TODO: areaWanted needs to be tested. 
"""
description: 
- approximately determines the greatest quadrilateral with point as one of the four corner points. 
  Given the start `point`, determines the extreme x and extreme y point using `incrementDirections`. 

arguments:
- point: iterable, length 2
- unqualifiedAreas: np.ndarray, n x 2 x 2
- area: np.ndarray, 2 x 2
- incrementDirections: dict, (increment::float), (direction::((0&1)::(1|-1)))
- areaWanted: max|xmax|ymax
"""
def greatest_qualifying_area_in_direction(point, unqualifiedAreas, area, incrementDirections, areaWanted):

    assert type(incrementDirections["increment"]) is float and incrementDirections["increment"] > 0, "inc. wrong type" 
    assert incrementDirections["direction"][0] in [-1,1], "axis 0 inc. wrong" 
    assert incrementDirections["direction"][1] in [-1,1], "axis 1 inc. wrong" 

    # extreme x
    id1 = {"axis": 0} 
    id1["increment"] = incrementDirections["increment"] * incrementDirections["direction"][0]
    pointX = greatest_qualifying_fourway_point_at_point(point, unqualifiedAreas, area, id1)

    # extreme y
    id2 = {"axis": 1} 
    id2["increment"] = incrementDirections["increment"] * incrementDirections["direction"][1]
    pointY = greatest_qualifying_fourway_point_at_point(point, unqualifiedAreas, area, id2)

    # TODO: refactor below. 
    if areaWanted == "max": 
        # get other extremes
        maxYGivenX = greatest_qualifying_fourway_point_at_point(pointX, unqualifiedAreas, area, id2) 
        maxXGivenY = greatest_qualifying_fourway_point_at_point(pointY, unqualifiedAreas, area, id1) 

        # get areas for each
        #   [0] 
        mp1 = missing_area_point_for_three_points(np.array([maxYGivenX, pointX, point]))
        area1 = trim_area(area, corners_to_area(np.array([maxYGivenX, pointX, point, mp1])))
        a1 = value_of_area(area1)
        #   [1] 
        mp2 = missing_area_point_for_three_points(np.array([maxXGivenY, pointY, point]))
        area2 = trim_area(area, corners_to_area(np.array([maxXGivenY, pointY, point, mp2])))

        a2 = value_of_area(area2)

        if a1 > a2: return area1
        return area2

    elif areaWanted == "xmax": 
        maxYGivenX = greatest_qualifying_fourway_point_at_point(pointX, unqualifiedAreas, area, id2)
        mp1 = missing_area_point_for_three_points(np.array([maxYGivenX, pointX, point]))
        area1 = trim_area(area, corners_to_area(np.array([maxYGivenX, pointX, point, mp1])))
        return area1 

    elif areaWanted == "ymax": 
        maxXGivenY = greatest_qualifying_fourway_point_at_point(pointY, unqualifiedAreas, area, id1) 
        mp2 = missing_area_point_for_three_points(np.array([maxXGivenY, pointY, point]))
        area2 = trim_area(area, corners_to_area(np.array([maxXGivenY, pointY, point, mp2]))) 
        return area2
    raise ValueError("invalid wanted area arg.")

# TODO: problem: what if want x > y or y < x ? 

# TODO: code this. delete above. 

"""
iterate from min x (point) in `incrementDirections` at in

incrementDirections: increment|axis
"""
def greatest_qualifying_fourway_point_at_point(point, unqualifiedAreas, area, incrementDirections): 

    if incrementDirections["increment"] >= 0:  
        term = lambda s: True if point >= area[1,incrementDirections["axis"]] else False 
    else:
        ##assert not (increment < 0), "invalid increment"
        term = lambda s: True if point <= area[1,incrementDirections["axis"]] else False 

    inc = lambda p: (p[0], p[incrementDirections["axis"]] + incrementDirections["increment"]) if incrementDirections["axis"] == 1 else\
                (p[incrementDirections["axis"]] + incrementDirections["increment"], p[1])

    prevPoint = None 
    point_ = (point[0], point[1])
    while point_in_area(point_, area): 
        # check for qual
        if not is_point_qualified(unqualifiedAreas, point_): break
        prevPoint = point_ 
        point_ = inc(point_) 
    return prevPoint

"""
description: 
- given a 4 x 2 matrix with each element a corner, determines the lower left and upper
  right points. 

arguments: 
- fourSizedArea: 4 x 2 np.ndarray

return: 
- 2 x 2 np.ndarray 
"""
def corners_to_area(fourSizedArea):
    assert not (len(fourSizedArea.shape) != 2 or fourSizedArea.shape[0] != 4\
        or fourSizedArea.shape[1] != 2)

    # get min x, min y
    minXIndices, minYIndices = np.where(fourSizedArea[:,0] == np.min(fourSizedArea[:,0])),\
                            np.where(fourSizedArea[:,1] == np.min(fourSizedArea[:,1]))

    index = np.intersect1d(minXIndices, minYIndices)
    if len(index) != 1: raise ValueError("points do not compose an area")    
    minPoint = fourSizedArea[index[0]]

    #   frequency count, minPoint[0] occurs twice, minPoint[1] occurs twice
    locs = np.where(fourSizedArea[:,0] == minPoint[0])[0] 
    if len(locs) != 2: raise ValueError("x coord. violation 1") 

    locs = np.where(fourSizedArea[:,1] == minPoint[1])[0]
    if len(locs) != 2: raise ValueError("y coord. violation 1") 

    # get max x, max y
    maxXIndices, maxYIndices = np.where(fourSizedArea[:,0] == np.max(fourSizedArea[:,0])),\
                            np.where(fourSizedArea[:,1] == np.max(fourSizedArea[:,1]))
    index = np.intersect1d(maxXIndices, maxYIndices)
    if len(index) != 1: raise ValueError("points do not compose an area")    
    maxPoint = fourSizedArea[index[0]] 

    #   frequency count, maxPoint[0] occurs twice, maxPoint[1] occurs twice
    locs = np.where(fourSizedArea[:,0] == maxPoint[0])[0] 
    if len(locs) != 2: raise ValueError("x coord. violation 1") 

    locs = np.where(fourSizedArea[:,1] == maxPoint[1])[0] 
    if len(locs) != 2: raise ValueError("y coord. violation 1") 
    return np.array([minPoint, maxPoint])

"""
"""
def trim_area(totalArea, area):
    assert is_valid_area(totalArea), "invalid total area"
    assert is_valid_area(area), "invalid area"
    # check min x
        # case: less than min x total 
    if area[0,0] < totalArea[0,0]: 
        area[0,0] = totalArea[0,0]

        # case: greater than max x total 
    elif area[0,0] > totalArea[1,0]: 
        area[0,0] = totalArea[1,0] 

    # check min y 
        # case: less than min y total 
    if area[0,1] < totalArea[0,1]: 
        area[0,1] = totalArea[0,1] 

        # case: greater than max y total 
    elif area[0,1] > totalArea[1,1]: 
        area[0,1] = totalArea[1,1] 

    # check max x
        # case: greater than max x total 
    if area[1,0] > totalArea[1,0]: 
        area[1,0] = totalArea[1,0] 

        # case: less than min x total
    elif area[1,0] < totalArea[0,0]: 
        area[1,0] = totalArea[0,0]  

    # check max y 
        # case: greater than max y total 
    if area[1,1] > totalArea[1,1]: 
        area[1,1] = totalArea[1,1] 

        # case: less than min y total
    elif area[1,1] < totalArea[0,1]: 
        area[1,1] = totalArea[0,1]
    return area

# TODO: matrix index accessor (using list of int.)
# TODO: approximate decimals to 5 places. 

# TODO: needs to be tested.

# TODO: does not perform argument check on three points!
"""
description: 
- determines the missing rectangular area point given three points. 

arguments: 
- threePoints: numpy array. 

return: 
- (float,float) 
"""
def missing_area_point_for_three_points(threePoints):
    assert is_2dmatrix(threePoints)

    # find the center point
    index = center_point(threePoints) 
    if index == -1: raise ValueError("[0] points given is invalid area") 
    center = threePoints[index] 

    # find the extreme-y point
    index2 = np.where(threePoints[:,0] == center[0])[0] 
    if len(index2) != 2: raise ValueError("[1] points given is invalid area") 
    yPoint = threePoints[index2[0] if index2[0] != index else index2[1]] 

    # find the extreme-x point
    index3 = np.where(threePoints[:,1] == center[1])[0] 
    if len(index3) != 2: raise ValueError("[2] points given is invalid area") 
    xPoint = threePoints[index3[0] if index3[0] != index else index3[1]]

    return (xPoint[0], yPoint[1])

# TODO: untested 
def other_points_for_two_points_in_area(twoPoints): 
    assert is_2dmatrix(twoPoints), "invalid points" 
    assert twoPoints.shape[0] == 2 and twoPoints.shape[1] == 2, "invalid shape for points"

    # quads 
    # 0 -> (max,max)
    # 1 -> (min,max) 
    # 2 -> (min,min) 
    # 3 -> (max,min) 
    ## can only obtain 1 and 3
    assert abs(twoPoints[0,0] - twoPoints[1,0]) > 10 ** -5, "two points cannot have same x-coord."
    assert abs(twoPoints[0,1] - twoPoints[1,1]) > 10 ** -5, "two points cannot have same y-coord."

    # get 1
    p1 = (min(twoPoints[:,0]), max(twoPoints[:,1]))

    # get 3
    p3 = (max(twoPoints[:,0]), min(twoPoints[:,1]))

    return (p1,p3)

# TODO: untested 
def two_points_to_area(twoPoints): 
    op = other_points_for_two_points_in_area(twoPoints) 
    data = np.vstack((twoPoints, op)) 
    return corners_to_area(data)

def diagonal_line_from_corner_point_in_area(area, cornerPoint): 
    q = quadrant_of_corner_point_in_area(area,cornerPoint)
    q2 = (q + 2) % 4
    return Line(np.array([q,q2]))
    
"""
description: 
- given three points that form a right angle, determines the point in the middle. 
  If the three points do not form a right angle, then returns -1 . 
"""
def center_point(threePoints:np.ndarray, index = 0):
    if index < 0: raise ValueError("invalid index")  
    if index >= threePoints.shape[0]: return -1

    indices = np.where(threePoints[:,0] == threePoints[index,0])[0] # TODO: check that equality works 
    indices2 = np.where(threePoints[:,1] == threePoints[index,1])[0] 
    if (True if len(indices) == 2 and len(indices2) == 2 else False): 
        return index
    return center_point(threePoints, index + 1)

def area_to_corners(area):
    assert is_valid_area(area), "invalid area {}".format(area)  
    upperLeft = [area[0,0], area[1,1]]
    lowerRight = [area[1,0], area[0,1]] 
    return np.round(np.array([area[1], upperLeft, area[0], lowerRight]),5)

############################################ START: qualifying area search

"""
return:
- (start)::float,(end)::float,(distance)::float
"""
def largest_subrange_of_coincidence_between_ranges(r1,r2, roundDepth = 5, hop = 0.05): 
    assert hop > 0, "hop > 0!"

    # sort
    assert len(r1) == 2 and len(r2) == 2, "invalid ranges {} and {}".format(r1,r2)
    r1,r2 = sorted(r1), sorted(r2)

    coin = False 
    longest = 0.0
    start, end = float('inf'), float('inf')
    now = 0.0
    startN, endN = float('inf'), float('inf')

    q = r1[0]
    while q <= r1[1]: 
        # coincides
        if q >= r2[0] and q <= r2[1]:        
            # start coincidence
            if not coin: 
                coin = True
                now = 0.0
                startN = q
                endN = float('inf')
                
            # update coincidence
            else: 
                now += hop 
                endN = q 
        
        # no longer coincides
        else:
            if coin: 
                coin = False
                if now > longest: 
                    longest = now
                    start,end = startN, endN 
                
                startN,endN = float('inf'), float('inf')

        q = round(q + hop, roundDepth)

    # update longest
    if now > longest: 
        longest = now
        start,end = startN, endN 
        
    return round(start, roundDepth) , round(end, roundDepth), round(longest, roundDepth)


def is_proper_color(color): 
    if color == None or type(color) in [int,float]: return False
    if len(color) != 3: return False #, "invalid length {} for color".format(len(color))
    for c in color: 
        if type(c) not in [int,float]: return False #, "invalid type {} for color".format(type(c))
    return True 

def area_to_pygame_rect_format(area):
    assert is_valid_area(area), "area {} is not valid!".format(area)
    lt = area[0]
    w,h = area[1,0] - area[0,0], area[1,1] - area[0,1] 
    return np.array([lt, [w,h]])

## TODO: 
"""
def area_from_point(p, areaWidth, areaHeight): 
    q2 = (p[0] + areaWidth, p[1] + areaHeight)
    return np.array([p,q2])
""" 

def point_to_area(p, areaWidth, areaHeight): 
    assert is_valid_point(p), "invalid point {}".format(p)
    assert areaWidth > 0.0 and areaHeight > 0.0, "invalid width and height"
    w,h = areaWidth / 2.0, areaHeight / 2.0 
    start = (p[0] - w, p[1] - h) 
    end = (p[0] + w, p[1] + h)
    return np.array([start,end])



################################## START: cast functions

def float_func(s): 
    q = float(s)
    return np.round(q, 5)

"""
always floor 
"""
#

def int_func(s):
    return int(floorceil_to_n_places(float(s), 'f', places = 5)) 

def floorceil_to_n_places(f, mode, places = 5): 
    assert places >= 0 and type(places) is int, "invalid places {}".format(places)
    assert mode in ["f","c"], "invalid mode {}".format(mode)

    d = f % 1.0
    w = f // 1.0 
    q = 0.0 
    lastDelta = 0.0 
    for i in range(places): 
        ten = 10 ** (-(i + 1))
        newD = d % ten
        q += (d - newD) 
        if (d - newD) > 0: lastDelta = ten
        d = newD
    o = w + q
    if mode == "f": return o 
    if f - o > 0: 
        return o + lastDelta

################################## END: cast functions

# 

# TODO: rename to `iterable`
def vector_to_string(v, castFunc = int):
    assert castFunc in [int,float, float_func], "invalid cast func"
    if len(v) == 0: return ""

    s = ""
    for v_ in v: 
        s += str(castFunc(v_)) + "," 
    return s[:-1] 

# TODO: untested 
def string_to_vector(s, castFunc = int):  
    assert castFunc in [float, float_func, int_func], "invalid cast func"

    def next_item(s):
        indo = len(s)  
        for (i, q) in enumerate(s): 
            if q == ",": 
                indo = i
                break 
        return s[:indo], s[indo + 1:] 
    
    q = [] 
    while s != "":
        s1,s2 = next_item(s) 
        v = castFunc(s1)
        q.append(v)     
        s = s2 
    return np.array(q)
