from measures import *
from globalls import * 

"""
line/line segment class
"""
class Line: 

    def __init__(self, endpoints, calculateCenter = True):
        endpoints = np.array(endpoints) 
        assert len(endpoints.shape) == 2 and endpoints.shape[0] == 2 and endpoints.shape[0] == endpoints.shape[1], "invalid endpoints"
        assert len(endpoints) == 2, "invalid number of endpoints"
        self.endpoints = endpoints
        self.gather_variables(calculateCenter) 
        self.data = None

    def __str__(self): 
        return vector_to_string(self.endpoints[0], float_func) + "|" + vector_to_string(self.endpoints[1], float_func)

    def __eq__(self, l2): 
        if l2 == None: return False 
        return equal_iterables(self.endpoints, l2.endpoints)

    # TODO: not tested.
    """
    partitions into n parts by axis with greatest change 
    """
    def n_partition(self,n, orderedByEnds = False):
        assert n > 1 and type(n) is int, "invalid n" 
        
        g = self.axis_with_greater_change()
        d = abs(self.endpoints[0,g] - self.endpoints[1,g])
        q = d / float(n)
        s = min([self.endpoints[0,g], self.endpoints[1,g]]) 
        
        r = np.empty((n + 1, 2))
        for i in range(n + 1): 
            x = s + (q * i)
            if g == 0: 
                y = self.y_given_x(x) 
                r[i] = [x,y] 
            else: 
                y = self.x_given_y(x)  
                r[i] = [y,x] 

        if orderedByEnds:
            if not equal_iterables(r[0], self.endpoints[0]): 
                return r[::-1]
        return r

    # TODO: not tested. 
    @staticmethod 
    def max_mod_distance_of_points_in_area(area, points, axisOfMod): 
        assert is_valid_area(area), "invalid area" 
        assert is_2dmatrix(points), "invalid points" 
        assert axisOfMod in [0,1], "invalid axis of mod."
        
        q = np.empty((points.shape[0], 2))
        for i in range(points.shape[0]): 
            assert point_in_area(tuple(points[i]), area), "point {} not in area\n{}".format(points[i],area)
            minDistance = abs(area[0,axisOfMod] - points[i,axisOfMod])
            maxDistance = abs(area[1,axisOfMod] - points[i,axisOfMod])
            q[i] = [minDistance, maxDistance] 
        return q

    # TODO: check intersection on T and L shapes.
    # TODO: not tested. 
    def crosses_area_edge(self,area, dualSidedCross = True): 
        assert is_valid_area(area), "invalid area" 

        c = area_to_corners(area) 
        
        for i in range(4):
            l = Line(np.array([c[i], c[(i + 1) %4]]))
            p, stat = intersection_between_two_lines(l, self)
            if stat:
                if dualSidedCross: 
                    if l.is_vertical(): 
                        if p[0] > self.endpoints[0,0] and p[0] < self.endpoints[1,0]:
                            return p,i
                    else: # h
                        if p[1] > self.endpoints[0,1] and p[1] < self.endpoints[1,1]:
                            return p,i
                else:
                    return p, i
        
        return None,-1

    def axis_with_greater_change(self):
        d0,d1 = abs(self.endpoints[0,0] - self.endpoints[1,0]), abs(self.endpoints[0,1] - self.endpoints[1,1]) 
        return 0 if d0 > d1 else 1

    def is_horizontal(self): 
        return True if self.slope <= 10 ** -5\
        and abs(self.endpoints[0,1] - self.endpoints[1,1])\
        < 10 ** -4 else False 

    def is_vertical(self): 
        return True if self.slope <= 10 ** -5\
        and abs(self.endpoints[0,0] - self.endpoints[1,0])\
        < 10 ** -4 else False 

    def change_in_xcoord_by_distance(self,distance):
        # TODO: ? sign restriction?  
        assert distance >= 0, "distance cannot be negative"
        distanceDeltaPerX = math.sqrt(self.slope ** 2 + 1)
        return distance / distanceDeltaPerX

    # TODO: floor round 
    def trim_from_end(self, endpoint, trimLength): 
        index = -1
        if equal_iterables(endpoint,self.endpoints[0]):
            index = 0 
        elif equal_iterables(endpoint,self.endpoints[1]):
            index = 1
        else:
            raise ValueError("endpoint {} not on this line".format(endpoint))

        # no change 
        if trimLength == 0.0: 
            return Line(np.copy(self.endpoints))

        assert trimLength > 0, "trim length must be pos."

        dx = self.change_in_xcoord_by_distance(trimLength) 

        # case: line got deleted
        if dx > point_distance(self.endpoints[0],self.endpoints[1]):
            return None

        # get the new x 
        otherIndex = 1 if index == 0 else 0 
        if self.endpoints[otherIndex,0] <= self.endpoints[index,0]: 
            dx = -dx
        
        newX = floorceil_to_n_places(endpoint[0] + dx, "f", 5) 
        newY = floorceil_to_n_places(self.y_given_x(newX), "f", 5)

        if index == 0: 
            new0,new1 = [newX,newY], np.copy(self.endpoints[otherIndex]) 
        else: 
            new0,new1 = np.copy(self.endpoints[otherIndex]), [newX,newY]

        return Line(np.array([new0,new1]))

    # TODO: important!, add round_info (floor/ceil)
    """
    description:
    - line gets trimmed from opposing end to start point
    """
    def trim_to_required_length_from_startpoint(self, requiredLength, endpoint):
        pd = point_distance(self.endpoints[0],self.endpoints[1])
        trimLength = pd - requiredLength
        assert trimLength > 0, "required length greater than length!"
        otherIndex = 1 if equal_iterables(endpoint, self.endpoints[0]) else 0
        return self.trim_from_end(self.endpoints[otherIndex], trimLength)

    # TODO: test this 
    def trim_to_fit_in_area(self,area):
        assert is_valid_area(area), "area invalid" 
        if not self.intersects_with_area(area): return None
        np0, np1 = np.copy(self.endpoints[0]), np.copy(self.endpoints[1])

        def check_endpoint_for_bounds_violation(e): 
            # case: x out of range
            if e[0] < area[0,0]: 
                newY = self.y_given_x(area[0,0]) 
                e = np.array([area[0,0], newY]) 
            elif e[0] > area[1,0]: 
                newY = self.y_given_x(area[1,0]) 
                e = np.array([area[1,0], newY])

            # case: y out of range 
            if e[1] < area[0,1]:
                newX = self.x_given_y(area[0,1])
                e = np.array([newX, area[0,1]])
            if e[1] > area[1,1]:
                newX = self.x_given_y(area[1,1])
                e = np.array([newX, area[1,1]]) 
            return e 

        np0,np1 = check_endpoint_for_bounds_violation(np0), check_endpoint_for_bounds_violation(np1)
        return Line(np.array([np0,np1]))  

    def trim_to_not_fit_in_area(self, area):
        q = self.trim_to_fit_in_area(area)
        if q == None: return [self] 

        cs = complement_of_range_in_range(tuple(self.endpoints[:,0]), tuple(q.endpoints[:,0]))
        ls_ = [] 
        for c in cs: 
            p1 = (c[0], self.y_given_x(c[0])) 
            p2 = (c[1], self.y_given_x(c[1])) 
            p = Line(np.array([p1,p2])) 
            ls_.append(p)  

        return ls_ 

    def union_with_bounds(self, area):
        xBound = (min(self.endpoints[0,0], self.endpoints[1,0]), max(self.endpoints[0,0], self.endpoints[1,0]))  
        yBound = (min(self.endpoints[0,1], self.endpoints[1,1]), max(self.endpoints[0,1], self.endpoints[1,1]))
        
        xInt = range_intersection(xBound, (area[0,0], area[1,0]))
        yInt = range_intersection(yBound, (area[0,1], area[1,1])) 
        return xInt and yInt

    '''
    line either 
    - exists completely in triangle
    - intersects w/ triangle edge 
    ''' 
    def point_in_triangle(self, a,b,c):

        # case: completely in triangle, check center pt. 
        if point_in_triangle(a,b,c,self.center): 
            return True 

        # case: intersection
        l_ = Line([a,b]) 
        _,stat = intersection_between_two_line_segments(self,l_)
        if stat: 
            return True 

        l_ = Line([a,c]) 
        _,stat = intersection_between_two_line_segments(self,l_)
        if stat: 
            return True 

        l_ = Line([b,c]) 
        _,stat = intersection_between_two_line_segments(self,l_)
        if stat: 
            return True 

        return False 

    '''
    similar to above, but w/ circle. 
    ''' 
    def point_in_circle(self, centerPoint, radius): 
        # case: completely in circle, check center pt. 
        if is_point_in_circle(self.center, centerPoint, radius):  
            return True 

        # case: intersects w/ circle, check both endpoints
        if is_point_in_circle(self.center, self.endpoints[0], radius):  
            return True 

        if is_point_in_circle(self.center, self.endpoints[1], radius):  
            return True

        return False 

    def intersects_with_area(self, area): 
        if not self.union_with_bounds(area): return False
        NONE,i = self.intersection_point_with_area(area)
        return True if i != -1 else False 

    # TODO: untested. 
    """
    return:
    - point
    - index::(edge) 
    """
    def intersection_point_with_area(self, area): 
        assert is_valid_area(area), "invalid area"
        # vertices
        """
        v1  v0 
        v2  v3
        """
        corners = area_to_corners(area)

        # make a copy        
        for i in range(4): 
            l = Line(np.array([corners[i], corners[(i + 1) %4]]))
            point, stat = intersection_between_two_line_segments(l, self)
            if stat: 
                return point,i
        return None,-1

    def subtract_line(self, l): 
        e1 = np.copy(self.endpoints[0])
        e2 = np.copy(l.endpoints[1])
        return Line(np.array(e1,e2))

    def cross_product_line(self, l): 
        assert equal_iterables(l.endpoints[0], self.endpoints[0]), "invalid iterables"

        # get angle b/t lines
        angle1 = angle_between_two_points_clockwise(self.endpoints[0], self.endpoints[1]) 
        angle2 = angle_between_two_points_clockwise(l.endpoints[0], l.endpoints[1]) 
        angle = min((angle1 - angle2) % 360, 360 - (angle1 - angle2) % 360)

        d1 = point_distance(self.endpoints[0], self.endpoints[1]) 
        d2 = point_distance(l.endpoints[0], l.endpoints[1]) 
        return d1 * d2 * math.sin(math.radians(angle)) 

    # TODO: unused. 
    def move_line_by_y_intercept(self, increment = 0.05):
        p1 = self.slope * self.endpoints[0,0] + self.yint + increment
        p2 = self.slope * self.endpoints[1,0] + self.yint + increment
        return Line([p1, p2])

    def move_line_by_xcoord(self, increment): 
        newEndpoints = np.array([[self.endpoints[0,0] + increment, self.endpoints[0,1]], [self.endpoints[1,0] + increment, self.endpoints[1,1]]])
        return Line(newEndpoints)

    def move_line_by_axis_shifts(self,shifts):
        assert is_valid_point(shifts), "invalid shifts"        
        np1 = (self.endpoints[0,0] + shifts[0], self.endpoints[0,1] + shifts[1])
        np2 = (self.endpoints[0,0] + shifts[0], self.endpoints[0,1] + shifts[1])
        return Line(np.array([np1,np2]))

    # TODO: test this
    def complement_of_subline_segment(self, l): 

        assert self.is_point_in_line_segment(l.endpoints[0])\
            and self.is_point_in_line_segment(l.endpoints[1]), "invalid subline {}".format(l) 

        r1 = np.copy(self.endpoints[:,0])
        r2 = np.copy(l.endpoints[:,0])

        complement = complement_of_range_in_range(r1, r2)
        lines = [] 

        for c in complement: 
            p1 = (c[0], self.y_given_x(c[0])) 
            p2 = (c[1], self.y_given_x(c[1])) 
            p = np.array([p1,p2])
            l2 = Line(p)
            lines.append(l2) 
        return lines

    # TODO: test this
    """
    default @ midpoint
    """
    def perpendicular_line(self, radius = 10, source = "mid"): 

        if source == "mid": 
            source = tuple(self.center)
        else:
            assert is_valid_point(source), "source is not point" 
            assert self.is_point_in_line_segment(source), "source is not in line segment" 
            
            y = self.y_given_x(sourceX) 
            source = (sourceX, y) 

        angle = angle_between_two_points_clockwise(self.endpoints[0],self.endpoints[1]) 

        neg = (angle - 90.0) % 360.0
        pos = (angle + 90.0) % 360.0

        q = generate_line_by_length_and_angle(source, radius, neg)
        q2 = generate_line_by_length_and_angle(source, radius, pos) 
        return Line(np.array([q.endpoints[1], q2.endpoints[1]]))

    def perpendicular_line_in_area_halfspace(self, halfspace): 
        assert halfspace in [-1,1], "invalid halfspace {}".format(halfspace)  
        assert is_valid_area(area), "invalid area" 
        mid = midpoint(self.endpoints[0], self.endpoints[1])

        degDiff = 90.0 * halfspace
        deg = angle_between_two_points_clockwise(self.endpoints[0], self.endpoints[1])

        # form area
        area = two_points_to_area(self.endpoints) 

        # get diagonal length total 
        diagLength = point_distance(area[0], area[1])
        boundX = area[0,0] if halfSpace < 0 else area[1,0]
        
        # draw perp line
        q = generate_line_by_length_and_angle(mid, diagLength, deg + degDiff) 
        return self.trim_to_fit_in_area(area)

    def gather_variables(self, calculateCenter):
        self.slope = point_slope(self.endpoints[0], self.endpoints[1])
        self.yint = self.endpoints[0][1] - self.endpoints[0][0] * self.slope
        self.center = None 
        if calculateCenter: 
            self.center = midpoint(self.endpoints[0], self.endpoints[1]) 

    # TODO: not tested. 
    def point_from_source_by_value(self, value, sourceIndex): 
        assert value >= 0 and value <= 1, "invalid value" 
        assert sourceIndex in [0,1], "invalid source index"
        l = value * point_distance(self.endpoints[0], self.endpoints[1]) 
        p = self.endpoints[sourceIndex]
        d = angle_between_two_points_clockwise(self.endpoints[0],self.endpoints[1]) 
        l2 = generate_line_by_length_and_angle(p,l,d)
        return l2.endpoints[1]

    # TODO: check for vertical lines 
    def y_given_x(self, x, rangeCheck = True):
        xs = [self.endpoints[0][0], self.endpoints[1][0]]
        if rangeCheck: 
            assert not (x < min(xs) or x > max(xs)), "x not in range of line" 
        return self.slope * x + self.yint

    # TODO: check for horizontal lines 
    def x_given_y(self, y, rangeCheck = True): 
        ys = [self.endpoints[0][1], self.endpoints[1][1]]
        if rangeCheck: 
            assert not (y < min(ys) or y > max(ys)), "y not in range of line" 

        if self.slope == 0: 
            return self.endpoints[0,0]
        return (y - self.yint) / float(self.slope)

    def is_point_in_line_segment(self, p):

        # case: horizontal line 
        if self.is_horizontal(): 
            return True if (p[1] - self.endpoints[0,1]) <= 10 ** -5 and\
                self.value_in_coordinate_range(p[0],0) else False 

        # case: vertical line 
        elif self.is_vertical(): 
            return True if (p[0] - self.endpoints[0,0]) <= 10 ** -5 and\
                self.value_in_coordinate_range(p[1],1) else False   

        # case: other 
        p2 = (p[0], self.y_given_x(p[0], False))

        return True if equal_iterables(p,p2) and\
            self.value_in_coordinate_range(p2[0], 0) and\
            self.value_in_coordinate_range(p2[1], 1) else False

    def value_in_coordinate_range(self, v, coord): 
        q = self.endpoints[:,coord]
        return True if v >= min(q) and v <= max(q) else False

    def form_point_sequence(self, output = "exact", pointHop = CR_DEFAULT_TRAVEL_HOP):## 0.1):
        assert output in ["integer", "exact"] 
        assert pointHop > 0, "invalid point hop"

        #### 
        """
        if self.endpoints[0][0] < self.endpoints[1][0]: 
            start, end = self.endpoints[0][0], self.endpoints[1][0]
        else: 
            start, end = self.endpoints[1][0], self.endpoints[0][0]

        self.data = [] 
        while start < end: 
            p = [int(start), int(self.y_given_x(start))] if output == "integer" else [start, self.y_given_x(start)] 
            self.data.append(p) 
            start += pointHop

        self.data = np.array(self.data)
        """ 

        #### 

        def get_point(v): 
            if a == 0: 
                return [int(v), int(self.y_given_x(start))] if output == "integer" else [start, self.y_given_x(start)] 
            else: 
                return [int(self.x_given_y(start)), int(v)] if output == "integer" else [self.x_given_y(start), start] 
        
        # determine axis of hop 
        a = self.axis_with_greater_change() 

        if self.endpoints[0][a] < self.endpoints[1][a]: 
            start, end = self.endpoints[0][a], self.endpoints[1][a]
        else: 
            start, end = self.endpoints[1][a], self.endpoints[0][a] 

        self.data = [] 

        while start < end: 
            self.data.append(get_point(start))  
            start += pointHop

        self.data = np.array(self.data)

    # TODO: check this. 
    """
    description: 
    - determines the (x,y) directions of line given its endpoints
      and slope. 
    """
    def coordinate_directions_given_endpoints(self): 
        xDir,yDir = None,None

        xDir = 1 if self.endpoints[0][0] <= self.endpoints[1][0] else -1
        yDir = 1 if self.slope >= 0 else -1
        return xDir, yDir

    # TODO: unused, delete. 
    def center_xcoord_at_zero(self):
        length = abs(self.endpoints[0,0] - self.endpoints[1,0]) 

        d = min(self.endpoints[:,0]) 
        sd = d + length / 2.0 
        newEndpoints = np.array([[self.endpoints[0,0] - sd, self.endpoints[0,1]], [self.endpoints[1,0] - sd, self.endpoints[1,1]] ])
        return Line(newEndpoints)


def intersection_between_two_lines(l1,l2): 

    # case: identical lines 
    if l1 == l2:
        return l1.endpoints[0], True 

    # case: one of the lines is vertical,
    v1,v2 = l1.is_vertical(), l2.is_vertical() 

    if v1 or v2: 
        if v1 and v2:
            return None,False

        xTarg = l1.endpoints[0,0] if v1 else l2.endpoints[0,0] 
        p = (xTarg, l1.y_given_x(xTarg, False)) if v2 else (xTarg, l2.y_given_x(xTarg, False))
        return p, True 


    # solve equation: m_1 * x + b = m_2 * x + b_2
    '''
    (m_1 - m_2) * x = b_2 - b 

    x = (b_2 - b) / (m_1 - m_2) 
    '''
    if abs(l1.slope - l2.slope) < 10 ** -5 and abs(l1.yint - l2.yint) > 10 ** -5: 
        return None, False 

    x = float(l2.yint - l1.yint) / float(l1.slope - l2.slope)
    y = l1.y_given_x(x, False) 
    return (x,y), True

def intersection_between_two_line_segments(l1,l2): 
    q, stat = intersection_between_two_lines(l1,l2)

    if not stat: return q,stat 
    if not l1.is_point_in_line_segment(q): return None, False 
    if not l2.is_point_in_line_segment(q): return None, False 
    return q,stat

# TODO: relocate. 
"""
"""
def point_in_triangle(a,b,c,p):
    if equal_iterables(a, p) or equal_iterables(b,p) or equal_iterables(c,p): 
        return True

    lab = Line(np.array([a,b]))
    lac = Line(np.array([a,c]))
    lap = Line(np.array([a,p]))
    abAng = angle_between_two_lines(lab,lac) 
    apAng = angle_between_two_lines(lab, lap) 
    
    if apAng > abAng:
        return False

    # get point of intersection b/t 
    lbc = Line(np.array([b,c])) 
    point,stat = intersection_between_two_lines(lbc, lap) 

    # check distance 
    apDist = point_distance(a, p) 
    aBoundDist = point_distance(a, point) 

    if apDist <= aBoundDist: return True 
    return False 
 
"""
"""
def angle_between_two_lines(l1, l2): 

    assert equal_iterables(l1.endpoints[0], l2.endpoints[0]), "invalid iterables"

    # get angle b/t lines
    angle1 = angle_between_two_points_clockwise(l1.endpoints[0], l1.endpoints[1]) 
    angle2 = angle_between_two_points_clockwise(l2.endpoints[0], l2.endpoints[1]) 
    angle = min((angle1 - angle2) % 360, 360 - (angle1 - angle2) % 360)
    return angle 

# TODO: un-used 
def tangent_of_line(line, point): 
    assert is_valid_point(point)

    # choose another x
    xp = point[0] + 10 
    q = y - (1 / line.slope * (xp - point[0]))  
    point2 = (xp, q)
    return Line(point, point2)