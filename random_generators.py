from line import *

################ START: line generators
"""
description:
- generates a random line of `length` from `startPoint` by a random angle in
  [0,360].
"""
def generate_random_line(startPoint, length):
    # random angle
    angle = random.uniform(0, 360)
    endPoint = hypotenuse_from_point(startPoint, length, angle)
    return Line((startPoint, endPoint))

def generate_random_line_at_center(centerPoint, length):
    # random angle
    angle = random.uniform(0, 360)
    altAngle = (angle + 180.0) % 360.0

    e1 = hypotenuse_from_point(centerPoint, length / 2.0, angle)
    e2 = hypotenuse_from_point(centerPoint, length / 2.0, altAngle)
    return Line((e1, e2))

################ END: line generators


    # TODO: CAUTION:
    ## relocate below lines to line.py for usage,
    ## non-circular dependencies.

    ################

def generate_line_by_length_and_angle(startPoint, length, angle):

    endPoint = hypotenuse_from_point(startPoint, length, angle)
    return Line((startPoint, endPoint))

"""
description:
- outputs an endpoint given `point`; `endpoint` is distance `length`
  from `point` and at `angle`
"""
def hypotenuse_from_point(point, length, angle):

    # get the x-delta and y-delta
    q = math.sin(math.radians(angle))
    opp = q * length

    q = math.cos(math.radians(angle))
    adj = q * length

    return [point[0] + adj, point[1] + opp]

    ##################################################################

# TODO: necessary??
"""
"""
def closest_right_angle_to_angle(angle):
    angle = angle % 360
    right = [0, 90, 180, 270, 360]
    diff = [abs(angle - r) for r in right]
    index = np.argmin(diff)
    return right[index]

"""
"""
def quadrant_of_angle(angle):

    assert not (angle < 0 or angle > 360), "invalid angle"

    if (angle >= 0 and angle <= 90) or angle == 360:
        return 0

    if (angle > 90 and angle <= 180):
        return 1

    if (angle > 180 and angle <= 270):
        return 2

    return 3

############################### END: line generator

################ START: area identification

def random_point_in_area(area):
    assert is_valid_area(area), "area is invalid"
    x = random.uniform(area[0,0], area[1,0])
    y = random.uniform(area[0,1], area[1,1])
    return (x,y)

def random_point_in_circle(center, radius):
    xDelta = random.uniform(-radius, radius)
    yDelta = random.uniform(-radius, radius)
    return (center[0] + xDelta, center[1] + yDelta)

"""
"""
def random_game_table_matrix(xMoveSize, yMoveSize, rangeX, rangeY):#, rule):
    assert len(rangeX) == 2 and len(rangeY) == 2, "arg. range is wrong"
    q = np.empty((xMoveSize, yMoveSize, 2))

    for i in range(xMoveSize):
        for j in range(yMoveSize):
            q[i,j] = (random.randrange(rangeX[0], rangeX[1]), random.randrange(rangeY[0], rangeY[1]))
    return q
