'''
basic polynomial solver by Lagrange basis functions
'''

from scipy import optimize as opt
from random_generators import * 
from travel_data import *
import logging

powerRow = lambda x, l: np.append(np.array([x ** (l - i) for i in range(l)]), np.array([1]))

# the effect of gravity at any instant, used to calculate acceleration
GRAVITY_AT_INSTANT = 11

class LagrangePolySolver: 
    
    def __init__(self, points, prefetch = True):  
        self.ps = points
        self.targetRangePoints = None
        self.vfs = None

        self.A = None

        self.recentRangeSearchInfo = None
        self.minumum, self.maximum = None,None
        self.integrl = None

        if prefetch: 
            self.prefetch_data()
        self.timeActionStamp = -1

        self.allTravelData = None # type TravelData 
            # variables used to declare allTravelData 
        self.velocityData = [] 
        self.durationData = []

    #
    @staticmethod
    def yield_lines_clockwise_from_source_in_area(sourcePoint, area, numParts):
        assert is_valid_area(area), "invalid area" 

        # check that source point is a corner of area 
        q = quadrant_of_corner_point_in_area(sourcePoint, area)
        if q == 2:
            dRange = [0,90]
        elif q == 3:
            dRange = [90,180]
        elif q == 0:
            dRange = [180,270]
        else:
            dRange = [270,360]

        degDelta = 90.0 / (numParts + 1) 
        xLength = area[1,0] - area[0,1]
        yLength = area[1,1] - area[0,1]

        maxLength = point_distance(area[0],area[1])
        for i in range(numParts):
            deg = dRange[0] + degDelta * (i + 1)
            l = generate_line_by_length_and_angle(sourcePoint, maxLength, deg)
            yield l.trim_to_fit_in_area(area)
    
    def prefetch_data(self): 
        self.bounds_for_x("float")
        self.integral_length(self.minumum, self.maximum, integrationHop = CR_DEFAULT_TRAVEL_HOP)   

    def display_basic(self): 
        print("\t* min: ", self.minumum, "\t* max: ", self.maximum)
        print("\t* points: ")
        print(self.ps) 
        print("\t* target range points: ")
        print(self.targetRangePoints) 
        print("\t* integral: ", self.integrl)

    ########################################## START: lagrange basis 

    """
    TODO: opt., make lambda function instead?? 
    """
    def lagrange_basis(self, x, pointIndex):  
        v = 1.0
        for i in range(len(self.ps)):
            if i == pointIndex: 
                continue 
            
            num = x - self.ps[i][0] 
            den = self.ps[pointIndex][0] - self.ps[i][0] 

            if den == 0: 
                #return 0 
                print("NOT GOOD: ", self.ps[pointIndex][0], " X2 ", self.ps[i][0])
                #v = 0 
            v = v * float(num) / float(den)
        return v

    def bounds_for_x(self, returnType):
        self.minumum, self.maximum = np.min(self.ps[:, 0]), np.max(self.ps[:,0])
        if returnType == "float": 
            return self.minumum, self.maximum 
        
        return int(math.ceil(self.minumum)), int(math.floor(self.maximum))

    def output_by_lagrange_basis(self, x): 
        y = 0 
        for i in range(len(self.ps)): 
            y += self.ps[i][1] * self.lagrange_basis(x, i)
        return y 
    
    # TODO: this needs to be fixed for missed hop at end
    def output_range(self, rangeX, intRound = False):

        self.targetRangePoints = np.empty((0,2)) 
        i = rangeX[0]
        xs = [y[0] for y in self.ps]
        while i < rangeX[1]:
            p = np.array([[i,self.output_by_lagrange_basis(i)]]) 
            self.targetRangePoints = np.append(self.targetRangePoints, p, axis = 0) 
            i += 1
                
        if intRound: 
            self.targetRangePoints = np.round(self.targetRangePoints) 
            self.targetRangePoints = self.targetRangePoints.astype(int) 
    
    ########################################## END: lagrange basis 

    def x_point_exists(self, x): 
        for (i,q) in enumerate(self.ps):
            if q[0] == x:
                return i
        return -1

    # TODO: code and refactor, multi-processing
    def integral_length(self, startX, endX, integrationHop = CR_DEFAULT_TRAVEL_HOP):           
        # TODO: off-by-1 bug here
        assert integrationHop > 0, "invalid integration hop"
        integrationHop = integrationHop if startX < endX else -integrationHop
        self.integrl = None
        prevPoint = (startX, self.output_by_lagrange_basis(startX))

        integrationHop 
        tFunc = lambda x: True if x <= endX and x >= startX else False  

        d = 0 
        while tFunc(startX): 
            startX += integrationHop
            point = (startX, self.output_by_lagrange_basis(startX))
            d += point_distance(prevPoint, point) 
            prevPoint = point
        self.integrl = d
        return d

    """
    return: 
    - (end point)::(2-tuple)
    - (distance travelled)::(float) 
    - (>= wanted distance)::(bool)
    """
    def get_xrange_for_wanted_distance(self, startPoint, distance, integrationHop = CR_DEFAULT_TRAVEL_HOP):
        
        assert not (distance <= 0), "invalid wanted distance"

        pf = lambda x: True if x >= self.minumum and x <= self.maximum else False 
        df = lambda x: True if x >= distance else False 
        r, q, gw = self.traversal_loop(startPoint, integrationHop, pf, df) 
        
        # trim leftovers 
        if gw: 
            newHop = integrationHop / 10.0 
            excess = q - distance
            prevPoint = (r[0] - integrationHop, self.output_by_lagrange_basis(r[0] - integrationHop))
            dPrev = q - point_distance(r, prevPoint)
            wanted = distance - dPrev

            df2 = lambda x: True if x >= wanted else False 
            r2, q2, gw2 = self.traversal_loop(r[0] - integrationHop, newHop, pf, df2)
            return r2, dPrev + q2, gw2
        else: 
            return r, q, gw

    """
    arguments: 
    - startPoint: float, x-value  
    - integrationHop: float, +|-
    - pFunc: function(float), point termination 
    - dFunc: function(float), distance termination

    return: 
    - (end point)::(2-tuple)
    - (distance travelled)::(float) 
    - (>= wanted distance)::(bool)
    """
    def traversal_loop(self, startPoint, integrationHop, pFunc, dFunc):
        prevPoint, r = (startPoint, self.output_by_lagrange_basis(startPoint)),\
                    (startPoint, self.output_by_lagrange_basis(startPoint))
        totalPd = 0.0
        totalPd_ = totalPd 
        gotWanted = False
        
        # TODO: find the cause of bug and delete fix
        noChange = 0
        c = 0 
        while pFunc(prevPoint[0]):
            c += 1
            r = (prevPoint[0] + integrationHop, self.output_by_lagrange_basis(prevPoint[0] + integrationHop)) 
            totalPd_ = totalPd
            totalPd += point_distance(prevPoint, r)

            if dFunc(totalPd): 
                gotWanted = True 
                break 

            if tuple(r) == tuple(prevPoint): 
                noChange += 1 
                if noChange >=100: 
                    print("\n\n\tNO CHANGE\n\n")
                    break 
            else: 
                noChange = 0
                prevPoint = (r[0], r[1])

        if r[0] >= self.maximum or r[0] <= self.minumum: 
            prevPoint = (prevPoint[0] - integrationHop, self.output_by_lagrange_basis(prevPoint[0] - integrationHop))
            return prevPoint, totalPd_, gotWanted
        return r, totalPd, gotWanted

    def average_slope_on_range(self, startX, endX): 
        assert startX < endX, "invalid start and end x points"
    
        prevPoint = (startX, self.output_by_lagrange_basis(startX))
        d = 0
        # TODO: refactor this to #$#
        while startX < endX: 
            startX += integrationHop
            point = (startX, self.output_by_lagrange_basis(startX))
            d += point_distance(prevPoint, point) 
            prevPoint = point
        return d / float(endX - startX)

    ########################### START: vector-form solution 

    def get_vfs_vars(self): 
        # form A 
        xs = self.ps[:,0]
        self.A = np.empty((xs.shape[0], xs.shape[0]))
        for (i,x) in enumerate(xs):
            q = powerRow(x, xs.shape[0] - 1)
            self.A[i] = q
        
        # form b
        ys = self.ps[:,1]
        return self.A, ys

    def vector_form_solution(self):
        self.A, ys = self.get_vfs_vars()
        self.vfs = np.linalg.solve(self.A, ys)

    def output_by_vector_form(self, x): 
        q = powerRow(x, self.ps.shape[0] - 1)        
        return q.dot(self.vfs) 

    def distance_between_points(self, x1, x2): 

        p1, p2 = self.output_by_lagrange_basis()
        return point_distance(p1, p2)

    """
    description:
    - draws a line between two polynomial points and determines the minumum velocity to travel. 
      
      CAUTION: does not consider points in (min(x1,x2), max(x1,x2))

    arguments:
    - x1: float
    - x2: float

    return: 
    - 
    """
    def required_velocity_for_travel_of_range_angle_effect(self, x1, x2):

        p1,p2 = (x1, self.output_by_lagrange_basis(x1)), (x2, self.output_by_lagrange_basis(x2)) 
        d = point_distance(p1, p2) 
        acc = self.acceleration_between_points(p1,p2)
        if acc >= 0: return 0 

        q = acc * d
        accTotal = math.sqrt(abs(q)) 
        duration = abs(accTotal / acc )
        return d / duration

    """
    description: 
    - uses above method, considers all points in range by hop increment

    return: 
    - float  
    """
    def required_velocity_for_travel_of_range_angle_effect_by_hop(self, x1, x2, hop):
        assert hop > 0, "hop must be greater than 0"
        assert x1 >= self.minumum and x1 <= self.maximum, "invalid arg [0]" 
        assert x2 >= self.minumum and x2 <= self.maximum, "invalid arg [1]"

        hop = -hop if x2 > x1 else hop
        q1 = x2
        q2 = x2 + hop 
        tfunc = lambda x: True if x >= min(x1,x2) and x <= max(x1,x2) else False

        vmin = 0.0  
        while tfunc(q2):
            p1,p2 = (q2, self.output_by_lagrange_basis(q2)), (q1, self.output_by_lagrange_basis(q1)) 
            vmin = self.back_hop_for_velocity(p1, p2, vmin)
            q1 = q2 
            q2 = round(q2 + hop, 3)
        return vmin

    def back_hop_for_velocity(self, x1, x2, velocity): 
        a = self.acceleration_between_points(x1,x2)
        b = -velocity 
        c = a * point_distance(x1,x2)
        soln, stat = quadratic_formula_solve_for_reals(1.0,b,c)

        # case: pos. accel. 
        if not stat: 
            return velocity - a
        return max(soln)

    def is_travel_over_range_feasible(self, x1, x2, hop, velocity):
        x = self.required_velocity_for_travel_of_range_angle_effect_by_hop(x1, x2, hop)
        if velocity >= x: 
            return True
        return False        
            
    def acceleration_between_points(self, x1, x2):
        if x1[0] == x2[0] and x1[1] == x2[1]: return 0 

        a = angle_between_two_points(x1, x2)
        prop = - (a / 90.0)
        return GRAVITY_AT_INSTANT * prop
    
    '''
    travel time == 1 if dist(x1,x2) == bv and ang(x1,x2) == 0
    '''
    def velocity_between_points_by_angle_effect(self, x1, x2, baselineVelocity): 
        return baselineVelocity + self.acceleration_between_points(x1, x2)

    # TODO: needs to be tested.
    """
    description: 
    - 

    return: 
    - (current velocity)::float
    - (able to travel)::bool
    - (duration)::float
    """
    def velocity_and_duration_between_points_by_angle_effect(self, x1, x2, baselineVelocity, duration = 0.0, c = 0):

        c += 1

        if baselineVelocity <= 0 or tuple(x1) == tuple(x2): return baselineVelocity, False, duration

        # TODO: refactor below equality statement
        if x1[0] == x2[0] and x1[1] == x2[1]: 
            return baselineVelocity, True, duration
        distance = point_distance(x1,x2)

        # case: acceleration will be divided        
        if distance <= baselineVelocity:
            a = self.acceleration_between_points(x1,x2) * distance / baselineVelocity
            return baselineVelocity + a, True, duration + distance / baselineVelocity
        else: 
        # case: travel time b/t points > 1 unit
            x1_, d, stat = self.get_xrange_for_wanted_distance(x1[0], baselineVelocity, CR_DEFAULT_TRAVEL_HOP if x1[0] <= x2[0] else -CR_DEFAULT_TRAVEL_HOP)
            baselineVelocity += self.acceleration_between_points(x1, x1_) 

            if equal_iterables(x1, x1_, roundPlaces = 5):
                return baselineVelocity, False, duration

            return self.velocity_and_duration_between_points_by_angle_effect(x1_, x2, baselineVelocity, duration + 1.0, c + 1)

    def travel_time_between_points(self, x1, x2, velocity):
        return point_distance(x1,x2) / velocity

    """
    arguments: 
    - startX: float, x-value
    - duration: float, >= 0
    - baselineVelocity: float, >=
    - hop: float, small

    return:
    - TravelData, (halt restricted)::bool, (bounds reached)::bool
    """
    def travel_from_start_for_duration_under_angle_effect(self, startX, duration, baselineVelocity, hop): 
        # neg 
        if hop <= 0: 
            term = lambda x: True if x > self.maximum else False
        # non-neg
        else: 
            term = lambda x: True if x < self.minumum else False
        
        if baselineVelocity <= 0: 
            print("\tCANNOT TRAVEL ANYMORE")
            return None, True, False 

        rfunc = self.continue_travelling
        tfunc = lambda x: True if x >= duration else False 
        dur = [0.0]
        distance = 0.0
        prevPoint = (startX, self.output_by_lagrange_basis(startX))
        max100 = 10
        allPoints = [prevPoint]

        # travel for duration 
        endPoint, allPoints, dur, baselineVelocity1, numHops, haltRestricted, stat = self.run_travel_loop(prevPoint, allPoints, rfunc, tfunc, hop, baselineVelocity, dur, "travel time history")        
        hopRanges = [((0, numHops), hop)]

        # case: out-of-bounds, cut off leftovers
        if stat or haltRestricted: 
            self.timeActionStamp += 1
            return TravelData(idn = self.timeActionStamp, pointData = allPoints[:-1], durationData = dur[:-1], velocityData = baselineVelocity1[:-1], hopRanges = hopRanges), haltRestricted, stat

        # run another travel loop for remainders 
        leftovers = sum(dur) - duration
        newWantedDuration = dur[-1] - leftovers 

        newHop = hop / 10.0 
        prevPoint = allPoints[-2]
        newTFunc = lambda x: True if x >= newWantedDuration else False
        bulkDuration = sum(dur[:-1])
        bv = baselineVelocity1[-2] if len(baselineVelocity1) >= 2 else baselineVelocity
        endPoint, allPoints2, dur2, baselineVelocity2, numHops2, haltRestricted, stat = self.run_travel_loop(prevPoint, [], rfunc, newTFunc, newHop, bv, [], "travel time history")
        hopRanges = [((0, numHops), hop), ((numHops, numHops + numHops2), newHop)]

        self.timeActionStamp += 1 
        return TravelData(idn = self.timeActionStamp, pointData = np.append(allPoints[:-1], allPoints2, axis = 0), durationData = dur[:-1] + dur2,\
            velocityData = np.append(baselineVelocity1[:-1], baselineVelocity2[1:], axis = 0), hopRanges = hopRanges), haltRestricted, stat 

    """
    ceil

    arguments:
    - rfunc: continue_travelling termination
    - tfunc: time threshold termination

    return: 
    - (point)::2d-tuple
    - (all points)::(float | list(floats))
    - (duration)::(float | list(floats))
    - (velocity)::(float | list(floats))
    - (number of hops)::float
    - (halt restriction)::bool
    - (at end of poly.)::bool
    """
    def run_travel_loop(self, prevPoint, allPoints, rfunc, tfunc, hop, baselineVelocity:float, duration:list, returnType:str): 
        distance = 0.0
        numHops = 0
        velocities = [baselineVelocity] 
        
        point = prevPoint
        bFunc = lambda p: True if p[0] >= self.minumum and p[0] <= self.maximum else False 
        endByTFunc = False
        haltRestricted = False
        ldx = len(duration) 
        while bFunc(prevPoint): 
            point = (prevPoint[0] + hop, self.output_by_lagrange_basis(prevPoint[0] + hop))
            ld0 = len(duration)
            duration2, baselineVelocity, distance2, haltRestricted, durSat = self.one_hop_for_travel_info(point, prevPoint, baselineVelocity, duration, distance, rfunc, tfunc, returnType)
            
            if haltRestricted:
                break

            duration = duration2 
            distance = distance2

            allPoints.append(point)
            velocities.append(baselineVelocity)
            numHops += 1

            if tfunc(sum(duration)):
                endByTFunc = True
                break 
            prevPoint = point

        return (point, allPoints[-1], sum(duration), baselineVelocity, numHops, haltRestricted, not endByTFunc)\
            if returnType == "travel time" else (point, np.array(allPoints), duration, velocities, numHops, haltRestricted, not endByTFunc)
        
    # TODO: refactor this and replace above
    """
    return:    
    - (duration)::(float | list(floats))
    - (velocity)::float
    - (distance travelled)::float
    - (negative halt restriction)::bool
    - (duration satisfied)::bool
    """
    def one_hop_for_travel_info(self, point, prevPoint, previousVelocity, prevTravelTime, distanceTravelled, rfunc, tfunc, returnType):
        assert returnType in ("travel time history", "travel time"), "invalid return type"

        velocity, stat, t = self.velocity_and_duration_between_points_by_angle_effect(prevPoint, point, previousVelocity)

        # not able to travel
        if not stat: 
            haltRestricted = True
        else: 
            # ? negative halt restriction
            haltRestricted = not rfunc(t) 

        if haltRestricted: 
            return (sum(prevTravelTime), velocity, distanceTravelled, haltRestricted, False) if returnType == "travel time" else\
                (prevTravelTime, velocity, distanceTravelled, haltRestricted, False)

        prevTravelTime.append(t)
        distanceTravelled += point_distance(prevPoint, point)

        durSat = not tfunc(sum(prevTravelTime))
        return (sum(prevTravelTime), velocity, distanceTravelled, haltRestricted, durSat) if returnType == "travel time"\
            else (prevTravelTime, velocity, distanceTravelled, haltRestricted, durSat)

    """
    # TODO: can be used for agent decision input by query
    """
    def continue_travelling(self, travelTime, travelHistory = None, query = False):

        if query: 
            raise ValueError("query mode needs to be implemented.")
        else:
            return True if travelTime >= 0 else False
        
    ################################## START: estimation methods 

    ################################## END: estimation methods 

    # TODO: caution, does not check args. start and end 
    def y_extremes_in_xrange(self, start, end, hopIncrement = 0.01): 
        assert hopIncrement > 0, "invalid hop increment"
        if start > end: 
            start,end = end,start 

        ysMin, ysMax = math.inf, - math.inf 
        while start < end:
            y = self.output_by_lagrange_basis(start)

            if y > ysMax: 
                ysMax = y

            if y < ysMin: 
                ysMin = y 
                
            start += hopIncrement

        return ysMin, ysMax
            
    """
    arguments: 
    - refinement: float, [0,1] specifies the degree to partition the polynomial x-range. 
    - totalArea: 2 x 2 np.array
    - radius: int, direction up and down from every point of consideration in polynomial.
    - forward: bool, specifies forward or backward direction for x-range. 

    return: 
    - list(area)
    """
    def capture_polynomial_as_disqualifying_areas(self, refinement, totalArea, radius = 4, forward = True): 
        q = (self.maximum - self.minumum) * refinement
        s = self.minumum if forward else lps.maximum
        if not forward: q = -q 
        areas = [] 
        while s > self.minumum and s < self.maximum:
            e = s + q 

            if forward and e > self.maximum: 
                e = self.maximum

            elif not forward and e < self.minumum: 
                e = self.minimum 

            yMin, yMax = self.y_extremes_in_xrange(s,e)

            if yMin in [np.inf, -np.inf] or yMax in [np.inf, -np.inf]: 
                raise ValueError("could not capture polynomial.")

            # form the four corners to an area
            minPoint, maxPoint = np.array([min(s,e), yMin]), np.array([max(s,e), yMax]) 
            area = np.array([minPoint, maxPoint]) 
            areas.append(area) 

            # update 
            s = e

        return areas 

    ##### START: repr methods 
    """

    """
    def form_point_sequence(self, reverse = False, hopIncrement = CR_DEFAULT_TRAVEL_HOP, capture = True): 
        assert hopIncrement > 0, "hop increment has to be +" 
        start = self.minumum if not reverse else self.maximum
        hopIncrement = -hopIncrement if reverse else hopIncrement
        tFunc = lambda x: True if x >= self.minumum and x <= self.maximum else False 
        
        allPoints = [] 
        while tFunc(start): 
            p = [start, self.output_by_lagrange_basis(start)] 
            allPoints.append(p) 
            start += hopIncrement
        
        allPoints = np.array(allPoints)
        if capture: self.data = allPoints
        return allPoints

    def poly_to_velocity_duration_measures_in_hop_iteration(self, velocity, reverse = False, hopIncrement = CR_DEFAULT_TRAVEL_HOP):
        # collect velocity data 
        if type(self.data) is None:
            self.form_point_sequence(reverse, hopIncrement)

        self.velocityData = [] 
        self.durationData = [] 
        for i in range(self.data.shape[0] - 1): 
            velocity, stat, duration = self.velocity_and_duration_between_points_by_angle_effect(self.data[i], self.data[i + 1], velocity)
            self.velocityData.append(velocity) 
            self.durationData.append(duration)

            # last element of data is no travel
            if not stat:
                self.data = self.data[:i + 2] 
                break

    def poly_to_travel_data(self, velocity, travelDataIdn = 0): 
        if type(self.data) is None: 
            self.form_point_sequence()     
        self.poly_to_velocity_duration_measures_in_hop_iteration(velocity)
        hopRanges = [((0, len(self.velocityData)), CR_DEFAULT_TRAVEL_HOP)]
        self.allTravelData = TravelData(travelDataIdn, self.data, self.durationData, self.velocityData, hopRanges, velocity)
        self.allTravelData.velocity_to_pointwise_acceleration_data() 

    def clear_travel_data_variables(self):
        self.velocityData = []
        self.durationData = []

    # TODO: test this. 
    """
    """
    def intersection_with_line(self, line): 

        if self.vfs is None: 
            self.vector_form_solution() 

        def f(x): 
            s = self.output_by_vector_form(x)
            s -= (line.slope * x + line.yint)
            return s

        try: 
            r1 = opt.brentq(f, self.minumum, self.maximum) #opt.newton(f, (self.maximum - self.minumum) / 2)
            return r1, True 
        except: 
            return math.inf, False

    def intersection_with_line_v2(self,line):
        if self.vfs is None: 
            self.vector_form_solution()

        def f(x): 
            s = self.output_by_vector_form(x)
            s -= (line.slope * x + line.yint)
            return s

        # center self along opposing ends 
        lps2,mid = self.center_by_endpoints(1)
        l2 = line.move_line_by_axis_shifts((0,mid))
        return lps2.intersection_with_line(l2)

    ############################################################

    # TODO: method will replace above. 
    # TODO: wrong. 
    def center_axis_at_zero(self, axis): 
        assert axis in [0,1], "invalid axis {}".format(axis) 

        if axis == 0:
            mini,maxi = self.minumum, self.maximum
        else:
            mini, maxi = self.y_extremes_in_xrange(self.minumum, self.maximum, hopIncrement = 0.05)

       # get center
        mid = (maxi - mini ) /  2.0 
        dest = -mid
        reloc = - abs(mini - dest)
        newData = np.copy(self.ps) 

        for i in range(newData.shape[0]): 
            newData[i,axis] += reloc 
        return LagrangePolySolver(newData), reloc

    def center_at_zero(self):
        q,xs = self.center_axis_at_zero(0)
        q2,ys = q.center_axis_at_zero(1)
        return q2,xs,ys

    """
    centers endpoints along axis
    0 -> balanced along y-axis
    1 -> balanced along x-axis
    """
    def center_by_endpoints(self, axis): 
        assert axis in [0,1], "invalid axis"

        # calc. midpoint
        # set midpoint as 0 for axis and shift all other points
        #       by - midpoint
        mid = (self.ps[0,axis] + self.ps[1,axis]) / 2.0 

        q = np.copy(self.ps) 
        for i in range(q.shape[0]):
            q[i,axis] = q[i,axis] - mid
        return LagrangePolySolver(q), -mid 

    ##### END: repr methods