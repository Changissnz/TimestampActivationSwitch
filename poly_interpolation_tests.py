from poly_interpolation import *
import unittest

class TestLagrangePolySolverMethods(unittest.TestCase):

    def test__LagrangePolySolver_vector_form_solution(self):

        points = np.array([(1,2), (2,7), (4, -4)])
        lps = LagrangePolySolver(points)
        lps.vector_form_solution()
        xpoints = [1, 1.5, 2, 3, 3.5, 4]

        for x in xpoints:
            lb = lps.output_by_lagrange_basis(x)
            vf = lps.output_by_vector_form(x)
            self.assertTrue(abs(lb - vf < 10 ** -5))#, "unequal values by lagrange and vector solution"
        return

    def test__LagrangePolySolver_required_velocity_for_travel_of_range_angle_effect(self):

        ## TODO:
        p3 = np.array([[0, 5],\
                    [10, 121],\
                    [16, -89],\
                    [22, 34]])

        # print lps info here
        lps = LagrangePolySolver(p3)
        rfunc = lps.continue_travelling

        # iterate through all points
        lps.form_point_sequence()
        for i in range(1,lps.data.shape[0]):
            point,prevPoint = lps.data[i], lps.data[i - 1]
            tfunc = lambda x: True if x >= point[0] else False
            v = lps.required_velocity_for_travel_of_range_angle_effect(prevPoint[0], point[0])
            dur,vel,dist,negHalt,durSat = lps.one_hop_for_travel_info(point, prevPoint, v, [], 0.0, rfunc, tfunc, "travel time")
            self.assertTrue(round(vel,3) == 0)

        return

    def test__LagrangePolySolver_required_velocity_for_travel_of_range_angle_effect_by_hop(self):

        p3 = np.array([[0, 5],\
                    [10, 121],\
                    [16, -89],\
                    [22, 34]])

        # write method to visualize poly better?
            # print lps info here
        lps = LagrangePolySolver(p3)
        rfunc = lps.continue_travelling

        v = lps.required_velocity_for_travel_of_range_angle_effect_by_hop(0, 22, 0.05)
        T,stat1,stat2 = lps.travel_from_start_for_duration_under_angle_effect(0, 100, v, 0.05)#, "travel time history")
        self.assertTrue(T.velocityData[-1] < 0.5)#, "incorrect velocity"

    def test__LagrangePolySolver_back_hop_for_velocity(self):

        p3 = np.array([[0, 5],\
                    [10, 121],\
                    [16, -89],\
                    [22, 34]])

        # write method to visualize poly better?
            # print lps info here
        lps = LagrangePolySolver(p3)
        v = lps.required_velocity_for_travel_of_range_angle_effect(21.95, 22.0)

        p1,p2 = (21.95, lps.output_by_lagrange_basis(21.95)), (22, lps.output_by_lagrange_basis(22))
        v2 =  lps.back_hop_for_velocity(p1, p2, 0.0)

        self.assertTrue(abs(v - v2) < 10 ** -5)#, "incorrect velocity for back hop"
        return

    def test__LagrangePolySolver_travel_from_start_for_duration_under_angle_effect(self):

        # case 1: non-flat polynomial, wanted duration cannot be achieved
        baselineVelocity = 25
        duration = 1
        p1 = np.array([[1,2], [2,7], [4, -4]])
        lps = LagrangePolySolver(p1)
        lps.bounds_for_x("float")

        # hop forward from left end
        start = 1.0
        hop = 0.05

        T,stat1,stat2 = lps.travel_from_start_for_duration_under_angle_effect(start, duration, baselineVelocity, hop)#, "travel time history")
        ##T.display_basic()
        self.assertTrue(T.totalDuration <= duration)#, "[0] actual duration does not exceed wanted duration"

        # case 2: flat polynomial
        p2 = np.array([[1,2], [5,2]])
        lps = LagrangePolySolver(p2)
        lps.bounds_for_x("float")
        T,stat1,stat2 = lps.travel_from_start_for_duration_under_angle_effect(start, duration, baselineVelocity, hop)#, "travel time history")

        self.assertTrue(round(T.pointData[-1,0],5) == 5.0)#, "[1] end point {} is not 5.0".format(T.pointData[-1,0])
        self.assertTrue(np.all(np.round(T.velocityData, 3) == 25.0))#, "[1] incorrect velocity"

        # case 3: non-flat polynomial, wanted duration achieved
        p3 = np.array([[1,2], [5, 8], [13, 19], [21, 19], [34, 22]])
        lps = LagrangePolySolver(p3)
        lps.bounds_for_x("float")

        T,stat1,stat2 = lps.travel_from_start_for_duration_under_angle_effect(start, duration, baselineVelocity, hop)#, "travel time history")
        self.assertTrue(abs(T.totalDuration - 1.0) <= 10 ** -3)#, "[2] duration could not be achieved"

        # case 4: non-flat polynomial, halt restriction
        baselineVelocity = 7.0
        p4 = np.array([[1,2], [10, 9002], [20, 19002]])
        lps = LagrangePolySolver(p4)
        lps.bounds_for_x("float")

        T,stat1,stat2 = lps.travel_from_start_for_duration_under_angle_effect(start, duration, baselineVelocity, hop)#, "travel time history")
        # assert yourself


    def test__LagrangePolySolver_integral_length(self):

        # case 0: flat line for x in [2,9]
        points = np.array([[2, 6], [6,6], [9,6]])

        lps = LagrangePolySolver(points)
        lps.integral_length(2,9)
        print("integral length: ", lps.integrl)

        # polynomial cannot be straight line
        self.assertTrue(abs(lps.integrl - 7.0) < 0.25, "def. wrong integral length")

        # TODO: time-test here


    def test__LagrangePolySolver_travel_from_start_for_duration_under_angle_effect_case2(self):
        # CASE 0:
        points = np.array([[0, 5],\
                    [10, 121],\
                    [16, -89],\
                    [22, 34]])

        velocity = 10
        lps = LagrangePolySolver(points)
        T,  stat, stat2 = lps.travel_from_start_for_duration_under_angle_effect(0, 10, velocity, CR_DEFAULT_TRAVEL_HOP)
        self.assertTrue(T.pointData.shape[0] == 2, "case [0]: wrong number of hops")
        return

        # TODO: make assertion for below
        # CASE 1:
        points = [[200., 200.],\
            [205.51362776, 394.44128542],\
            [298.88029449, 405.49572502],\
            [337.12424895, 399.43944835],\
            [385.0893603, 261.42341642],\
            [439.36642168, 414.1732305 ]]
        points = np.array(points)
        lps = LagrangePolySolver(points)
        T,  stat, stat2 = lps.travel_from_start_for_duration_under_angle_effect(200, 15, 30, CR_DEFAULT_TRAVEL_HOP)#, "travel time history")
        print("T ")
        T.display_basic()
        print("\nLPS DATA")
        lps.display_basic()
        return

    def test__LagrangePolySolver_get_xrange_for_wanted_distance(self):

        # case [0]: straight line

        ##  subcase [0]
        velocity = 3
        points = np.array([[0, 4], [2,4], [7,4], [9,4]])
        lps = LagrangePolySolver(points)
        lps.bounds_for_x("float")
        e,d,stat = lps.get_xrange_for_wanted_distance(0.0, 5, integrationHop = 0.05)
        ##print('results: ', e, d, stat)
        self.assertTrue((abs(d - 5) < 0.05), "wrong value for case 0")

        ## subcase [1]: wanted distance greater than range
        e2,d2,stat2 = lps.get_xrange_for_wanted_distance(0.0, 12, integrationHop = 0.05)
        ##print('results: ', e2, d2, stat2)
        self.assertTrue(abs(d2 - 9) < 0.05, "wrong value for case 1")
        return


    def test__LagrangePolySolver_velocity_and_duration_between_points_by_angle_effect(self):

        ## CASE 0:
        p3 = np.array([[0, 5],\
                    [10, 121],\
                    [16, -89],\
                    [22, 34]])

        # write method to visualize poly better?
            # print lps info here
        lps = LagrangePolySolver(p3)

        point, prevPoint = (0.5, lps.output_by_lagrange_basis(0.5)), (0.0, lps.output_by_lagrange_basis(0.0))
        cv, travel,dur = lps.velocity_and_duration_between_points_by_angle_effect(prevPoint, point, 23, duration = 0.0, c = 0)
        self.assertTrue(not travel, "case [0]: cannot travel over range")

        # CASE 1:
        p4 = np.array([[0, 53],\
                    [16, 53]])
        velocity = 23

        lps = LagrangePolySolver(p4)
        point, prevPoint = (0.5, lps.output_by_lagrange_basis(0.5)), (0.0, lps.output_by_lagrange_basis(0.0))
        cv, travel,dur = lps.velocity_and_duration_between_points_by_angle_effect(prevPoint, point, velocity, duration = 0.0, c = 0)
        self.assertTrue(travel and abs(cv - 23.0) < 10 ** -3, "case [1]: incorrect travel results")

        # TODO: USE matplot to visualize velocity, duration variables


    def test__LagrangePolySolver__intersection_with_line(self):

        p = np.array([[120, 120], [160, 160], [250, 200]])
        lps = LagrangePolySolver(p)

        l = np.array([[130, 140], [170, 140]])
        line = Line(l)
        x,x2 = lps.intersection_with_line_v2(line)
        self.assertTrue(x2, "intersection exists b/t lps2 and l2")


    def test__LagrangePolySolver__yield_lines_clockwise_from_source_in_area(self):

        sourcePoint = np.array([50,50])
        area = np.array([[50,50], [175, 290]])

        parts = 3
        ls = LagrangePolySolver.yield_lines_clockwise_from_source_in_area(sourcePoint, area, parts)

        c = 0
        for l in ls:
            p,i = l.crosses_area_edge(area)
            self.assertTrue(i == -1, "line not in bounds")
            c += 1
        assert c == parts, "incorrect # of lines"
    


if __name__ == '__main__':
    unittest.main()
