# TODO: fix import errors
from measures import *
from matrix_methods import *
from collections import OrderedDict

# TODO: make assertion, point,duration,velocity data all same length
class TravelData:

    """
    arguments:
    - hopRanges: list::((int::min, int::max), float::hop)
    """
    def __init__(self, idn, pointData, durationData, velocityData, hopRanges, startVelocity = None):
        self.idn = idn
        self.pointData = pointData
        self.durationData = durationData
        self.cumulativeDurationData = np.cumsum(self.durationData)
        self.velocityData = velocityData
        self.startVelocity = startVelocity
        self.accelerationData = None
        self.hopRanges = hopRanges
        self.totalDuration = np.sum(self.durationData)

    def timestamp_range(self):
        return self.cumulativeDurationData[-1] - self.cumulativeDurationData[0]

    # TODO: does not consider hopRanges, and does not
    #       use a static time unit constant.
    def velocity_to_pointwise_acceleration_data(self):
        if len(self.velocityData) == 0: raise ValueError("no velocity data for p")

        self.accelerationData = []
        prev = self.startVelocity
        for i in range(len(self.velocityData)):
            acc = self.velocityData[i] - prev
            prev = self.velocityData[i]
            self.accelerationData.append(acc)

    """
    return:
    - float
    """
    def length_of_travel(self):
        hri = 0
        travelLength = 0.0
        for (i, d) in enumerate(self.durationData):
            if not (i >= self.hopRanges[hri][0] and\
                i <= self.hopRanges[hri][1]):
                hri += 1
                ##continue
            travelLength += self.hopRanges[hri][1]
        return travelLength

    """

    return:
    - point/duration/velocity data index, previous hop increment taken
    """
    def closest_data_indices_for_wanted_duration(self, wantedDuration):

        lesser = None
        more = None
        l = len(self.cumulativeDurationData) - 1

        for i in range(l):
            if self.cumulativeDurationData[i] - wantedDuration <= 0 and\
                self.cumulativeDurationData[i + 1] - wantedDuration >= 0:
                lesser = i
                more = i + 1
                break

        if lesser == None:
            ci = l
        else:
            ci = lesser if abs(self.cumulativeDurationData[lesser] - wantedDuration)\
                < abs(self.cumulativeDurationData[more] - wantedDuration) else more

        hopIncrement = -1
        for x in self.hopRanges:
            if x[1][0] <= ci and x[1][1] >= ci:
                hopIncrement = x[0]

        assert not (hopIncrement == -1), "hop increment data for wanted duration could not be found"
        return ci, hopIncrement

    """

    """
    def velocity_to_acceleration(self, capture = False):
        a =  np.array([self.velocityData[i] - self.velocityData[i - 1]\
            for i in range(1, len(self.velocityData))])
        if capture: self.accelerationData = a
        return a

    def display(self):
        print("TRAVEL DATA")
        print("** idn: ", self.idn)
        print("** pd:\n{}\n".format(self.pointData))
        print("** dd: \n{}\n".format(self.durationData))
        print("** vd: \n{}\n".format(self.velocityData))
        print("** ad: \n{}\n".format(self.accelerationData))
        print("** hr: \n{}\n".format(self.hopRanges))

    def display_basic(self):
        print("TRAVEL DATA")
        print("** idn: ", self.idn)
        print("** first point: {}\n** last point: {}".format(self.pointData[0] if len(self.pointData) > 0 else False, self.pointData[-1] if len(self.pointData) > 0 else False))
        print("** shape of points: ", self.pointData.shape)
        print("** total duration: ", sum(self.durationData))
        print("** number of duration units: ", len(self.durationData))
        print("** number of velocity units: ", len(self.velocityData))
        print("** ending velocity: ", self.velocityData[-1] if len(self.velocityData) > 0 else False)
        print("** hr: ", self.hopRanges)

    def yield_strings_csv_format(self, stringOrList):
        assert stringOrList in [str,list], "invalid output type"
        for (p,d,v) in zip(self.pointData, self.durationData, self.velocityData):
            q = "{},{},{}".format(p,d,v) if stringOrList is str else [p,d,v]
            yield "{},{},{}".format(p,d,v)
