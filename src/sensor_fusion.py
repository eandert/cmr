import math
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree
from shapely.geometry import Polygon
import bisect

import utils
from sensor import DetectedObject


max_id = 10000


class MatchClass:
    def __init__(self, id, x, y, covariance, dx, dy, d_confidence, confidence, trust_score, object_type, time, width, length, angle):
        self.x = x
        self.y = y
        self.covariance = covariance
        self.dx = dx
        self.dy = dy
        self.velocity_confidence = d_confidence
        self.type = object_type
        self.last_measurement = time
        self.id = id
        self.confidence = confidence
        self.trust_score = trust_score
        self.width = width
        self.length = length
        self.angle = angle


class ResizableKalman:
    def __init__(self, time, x, y, fusion_mode):
        # This list will take
        self.localTrackersHList = []
        self.localTrackersMeasurementList = []
        self.localTrackersCovarianceList = []
        self.localTrackersIDList = []

        # Init the covariance to some value
        self.error_covariance = np.array(
            [[1.0, 0.0], [0.0, 1.0]], dtype='float')
        self.d_covariance = np.array([[2.0, 0.0], [0.0, 2.0]], dtype='float')

        # Store the first value for matching purposes
        self.x = x
        self.y = y

        # Store the covariance vs. expected
        self.error_tracker_temp = []

        # Arbitrary min tracking size so we are not too small to match to
        self.min_size = .5

        # Process varaition guess
        process_variation = .16

        # Track the time of the last track
        self.last_measurement = time

        # Track the number of times this Kalman filter has been used
        self.idx = 0

        # The amount of time before a tracker is removed from the list
        self.time_until_removal = .5

        # Store the fusion mode
        self.fusion_mode = fusion_mode

        # Trupercept stuff
        self.trupercept_list = []

        process_noise_estimate = .125

        # Set up the Kalman filter
        # Initial State cov
        if self.fusion_mode == 0:
            # Set up the Kalman filter
            self.F_t_len = 4
            # Setup for x_hat = x + dx,  y_hat = y + dy
            # Initial State cov
            self.P_hat_t = np.identity(4)
            self.P_hat_t[2][2] = 0.0
            self.P_hat_t[3][3] = 0.0
            # Process cov
            four = process_variation * (process_noise_estimate*process_noise_estimate*process_noise_estimate*process_noise_estimate)/4.0
            three = process_variation * (process_noise_estimate*process_noise_estimate*process_noise_estimate)/3.0
            two = process_variation * (process_noise_estimate*process_noise_estimate)/2.0
            one = process_variation * (process_noise_estimate)
            self.Q_t = np.array([[three, 0, two, 0],
                                [0, three, 0, two],
                                [two, 0, process_variation, 0],
                                [0, two, 0, process_variation]], dtype='float')
            # Control matrix
            self.B_t = np.array([[0], [0], [0], [0]], dtype='float')
            # Control vector
            self.U_t = 0
        elif self.fusion_mode == 1:
            # Setup for x_hat = x + dx + dxdx,  y_hat = y + dy + dydy
            self.F_t_len = 6
            self.P_hat_t = np.identity(6)
            self.P_hat_t[2][2] = 0.0
            self.P_hat_t[3][3] = 0.0
            self.P_hat_t[4][4] = 0.0
            self.P_hat_t[5][5] = 0.0
            # Process cov
            five = process_variation * (process_noise_estimate*process_noise_estimate*process_noise_estimate*process_noise_estimate*process_noise_estimate)/8.0
            four = process_variation * (process_noise_estimate*process_noise_estimate*process_noise_estimate*process_noise_estimate)/4.0
            three = process_variation * (process_noise_estimate*process_noise_estimate*process_noise_estimate)/2.0
            two = process_variation * (process_noise_estimate*process_noise_estimate)
            self.Q_t = np.array([[five, 0, four, 0, 0, 0],
                                [0, five, 0, four, 0, 0],
                                [four, 0, three, 0, three, 0],
                                [0, four, 0, three, 0, three],
                                [0, 0, three, 0, two, 0],
                                [0, 0, 0, three, 0, two]], dtype='float')
            # Control matrix
            self.B_t = np.array([[0], [0], [0], [0], [0], [0]], dtype='float')
            # Control vector
            self.U_t = 0
        else:
            # model from https://journals.sagepub.com/doi/abs/10.1177/0959651820975523
            self.F_t_len = 5
            self.P_hat_t = np.identity(5)
            self.P_hat_t[2][2] = 0.0
            self.P_hat_t[3][3] = 0.0
            self.P_hat_t[4][4] = 0.0
            # Process cov
            four = process_variation * (process_noise_estimate*process_noise_estimate*process_noise_estimate*process_noise_estimate)/4.0
            three = process_variation * (process_noise_estimate*process_noise_estimate*process_noise_estimate)/2.0
            angle_variation = process_variation
            two = angle_variation * (process_noise_estimate*process_noise_estimate)
            self.Q_t = np.array([[four, 0, three, 0, 0],
                                [0, four, 0, three, 0],
                                [three, three, two, 0, 0],
                                [0, 0, 0, two, 0],
                                [0, 0, 0, 0, two]], dtype='float')
            # Control matrix
            self.B_t = np.array([[0], [0], [0], [0], [0]], dtype='float')
            # Control vector
            self.U_t = 0

    def addFrames(self, measurement_list):
        # try:
        # Rebuild the lists every time because measurements come and go
        self.localTrackersCovarianceList = []
        self.localTrackersMeasurementList = []
        self.localTrackersHList = []
        self.localTrackersIDList = []
        self.localTrackersExtraList = []
        # Check if there are more sensors in the area that have not been added
        for match in measurement_list:
            self.localTrackersIDList.append(match.id)
            self.localTrackersCovarianceList.append(match.covariance)
            # , match.dx, match.dy]))
            self.localTrackersMeasurementList.append(
                np.array([match.x, match.y]))
            self.localTrackersHList.append(0)
            self.localTrackersExtraList.append(
                [match.confidence, match.trust_score])

    def h_t(self, h_t_type):
        if h_t_type == 0:
            if self.fusion_mode == 0:
                return np.array([[1., 0., 0., 0.],
                                [0., 1., 0., 0.]], dtype='float')
            elif self.fusion_mode == 1:
                return np.array([[1., 0., 0., 0., 0., 0.],
                                [0., 1., 0., 0., 0., 0.]], dtype='float')
            elif self.fusion_mode == 2:
                return np.array([[1., 0., 0., 0., 0.],
                                [0., 1., 0., 0., 0.]], dtype='float')
        else:
            # TODO: implement radar type
            return np.array([[0, 0., 0., 0.],
                            [0., 0, 0., 0.]], dtype='float')

    def averageMeasurementsFirstFrame(self):
        if len(self.localTrackersCovarianceList) == 1:
            return self.localTrackersMeasurementList[0], self.localTrackersCovarianceList[0]

        for idx, cov in enumerate(self.localTrackersCovarianceList):
            if idx == 0:
                temporary_c = cov.transpose()
            else:
                temporary_c = np.add(temporary_c, cov.transpose())
        temporary_c = temporary_c.transpose()

        for idx, (pos, cov) in enumerate(zip(self.localTrackersMeasurementList, self.localTrackersCovarianceList)):
            if idx == 0:
                temporary_mu = np.matmul(cov.transpose(), pos)
            else:
                temporary_mu = np.add(
                    temporary_mu, np.matmul(cov.transpose(), pos))
        temporary_mu = np.matmul(temporary_c, temporary_mu)

        return temporary_mu, temporary_c

    def fusion(self, measurement_list, time, monitor):
        # Set the kalman variables and resize the arrays dynalically (if needed
        self.addFrames(measurement_list)
        # Do the kalman thing!
        if self.idx == 0:
            # We have no prior detection so we need to just output what we have but store for later
            # Do a Naive average to get the starting position
            pos, cov = self.averageMeasurementsFirstFrame()

            # Store so that next fusion is better
            self.x = pos[0]
            self.y = pos[1]
            self.error_covariance = cov
            self.idx += 1

            if self.fusion_mode == 0:
                # Store so that next fusion is better
                self.X_hat_t = np.array(
                    [[self.x], [self.y], [0], [0]], dtype='float')
            elif self.fusion_mode == 1:
                # setup for x, dx, dxdx
                self.X_hat_t = np.array(
                    [[self.x], [self.y], [0], [0], [0], [0]], dtype='float')
            else:
                # setup for https://journals.sagepub.com/doi/abs/10.1177/0959651820975523
                self.X_hat_t = np.array(
                    [[self.x], [self.y], [0], [0], [0]], dtype='float')

            # Seed the covariance values directly from the measurement
            self.P_hat_t[0][0] = self.error_covariance[0][0]
            self.P_hat_t[0][1] = self.error_covariance[0][1]
            self.P_hat_t[1][0] = self.error_covariance[1][0]
            self.P_hat_t[1][1] = self.error_covariance[1][1]
            self.last_update = time
            self.x = self.x
            self.y = self.y
            self.dx = 0.0
            self.dy = 0.0
            self.idx += 1
        else:
            # try:
            # We have valid data
            # Transition matrix
            elapsed = self.last_update - time
            if elapsed <= 0.0:
                # print( "Error time elapsed is incorrect! " + str(elapsed) )
                # Set to arbitrary time
                elapsed = 0.1

            if self.fusion_mode == 0:
                self.F_t = np.array([[1, 0, elapsed, 0],
                                    [0, 1, 0, elapsed],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], dtype='float')

            elif self.fusion_mode == 1:
                # setup for x, dx, dxdx
                self.F_t = np.array([[1, 0, elapsed, 0, elapsed*elapsed, 0],
                                    [0, 1, 0, elapsed, 0, elapsed*elapsed],
                                    [0, 0, 1, 0, elapsed, 0],
                                    [0, 0, 0, 1, 0, elapsed],
                                    [0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 1]], dtype='float')

            else:
                # setup for https://journals.sagepub.com/doi/abs/10.1177/0959651820975523
                v_1 = self.X_hat_t[2]
                phi_1 = self.X_hat_t[3]
                phi_dot_1 = self.X_hat_t[4]
                # print ( self.X_hat_t, v_1, phi_1, phi_dot_1 )
                if phi_1 == 0.0:
                    # Not moving, No correlation
                    v_x = 999.9
                    v_y = 999.9
                    phi_x = 999.9
                    phi_y = 999.9
                else:
                    v_x = (1.0 / phi_dot_1) * (-math.sin(phi_1) +
                                               math.sin(phi_1 + elapsed * phi_dot_1))
                    v_y = (1.0 / phi_dot_1) * (math.cos(phi_1) -
                                               math.cos(phi_1 + elapsed * phi_dot_1))
                    phi_x = (v_1 / phi_dot_1) * (-math.cos(phi_1) +
                                                 math.cos(phi_1 + elapsed * phi_dot_1))
                    phi_y = (v_1 / phi_dot_1) * (-math.sin(phi_1) +
                                                 math.sin(phi_1 + elapsed * phi_dot_1))
                if phi_dot_1 == 0.0:
                    # Not accelerating, NO correlation
                    phi_dot_x = 999.9
                    phi_dot_y = 999.9
                else:
                    phi_dot_x = (v_1 * elapsed / phi_dot_1) * math.cos(phi_1 + elapsed * phi_dot_1) - (
                        v_1 / phi_dot_1**2) * (- math.sin(phi_1) + math.sin(phi_1 + elapsed * phi_dot_1))
                    phi_dot_y = (v_1 * elapsed / phi_dot_1) * math.sin(phi_1 + elapsed * phi_dot_1) - (
                        v_1 / phi_dot_1**2) * (math.cos(phi_1) - math.cos(phi_1 + elapsed * phi_dot_1))
                self.F_t = np.array([[1, 0, v_x, phi_x, phi_dot_x],
                                    [0, 1, v_y, phi_y, phi_dot_y],
                                    [0, 0, 1, 0, 0],
                                    [0, 0, 0, 1, elapsed],
                                    [0, 0, 0, 0, 1]], dtype='float')

            self.X_hat_t, self.P_hat_t = utils.kalman_prediction(
                self.X_hat_t, self.P_hat_t, self.F_t, self.B_t, self.U_t, self.Q_t)

            added = 0
            if len(self.localTrackersMeasurementList) != 0:
                for mu, cov, h_t_type, extra in zip(self.localTrackersMeasurementList, self.localTrackersCovarianceList, self.localTrackersHList, self.localTrackersExtraList):
                    if extra[1] < 1.2:
                        Z_t = mu.transpose()
                        Z_t = Z_t.reshape(Z_t.shape[0], -1)
                        self.X_hat_t, self.P_hat_t = utils.kalman_update(
                            self.X_hat_t, self.P_hat_t, Z_t, cov.dot(extra[1]), self.h_t(h_t_type))
                        added += 1
            if len(self.localTrackersMeasurementList) == 0 or added == 0:
                nothing_cov = np.array([[1.0, 0.],
                                        [0., 1.0]], dtype='float')
                measure = np.array([.0, .0], dtype='float')
                if self.fusion_mode == 0:
                    nothing_Ht = np.array([[0, 0., 0., 0.],
                                           [0., 0, 0., 0.]], dtype='float')
                elif self.fusion_mode == 1:
                    nothing_Ht = np.array([[0, 0., 0., 0., 0., 0.],
                                           [0., 0., 0., 0., 0., 0.]], dtype='float')
                elif self.fusion_mode == 2:
                    nothing_Ht = np.array([[0, 0., 0., 0., 0.],
                                           [0., 0, 0., 0., 0.]], dtype='float')

                Z_t = (measure).transpose()
                Z_t = Z_t.reshape(Z_t.shape[0], -1)
                self.X_hat_t, self.P_hat_t = utils.kalman_update(
                    self.X_hat_t, self.P_hat_t, Z_t, nothing_cov, nothing_Ht)

            # Lets check the accuracy of each sensing platform
            self.error_tracker_temp = []
            length = len(self.localTrackersIDList)
            if monitor:
                # Our method
                if self.P_hat_t[0][0] != 0.0 and self.P_hat_t[0][1] != 0.0:
                    global_error_x, global_error_y, global_error_angle = utils.ellipsify(
                        [[self.P_hat_t[0][0], self.P_hat_t[0][1]], [self.P_hat_t[1][0], self.P_hat_t[1][1]]], 1.0)
                else:
                    global_error_x = 0.0
                    global_error_y = 0.0
                    global_error_angle = 0.0
                for id, mu, cov, h_t_type in zip(self.localTrackersIDList, self.localTrackersMeasurementList, self.localTrackersCovarianceList, self.localTrackersHList):
                    Z_t = (mu).transpose()
                    Z_t = Z_t.reshape(Z_t.shape[0], -1)
                    y_t_temp = Z_t - self.h_t(h_t_type).dot(self.X_hat_t)
                    # print(y_t_temp)
                    location_error = math.hypot(y_t_temp[0], y_t_temp[1])
                    expected_a, expected_b, expected_angle = utils.ellipsify(
                        cov, 1.0)
                    expected_x = utils.calculateRadiusAtAngle(
                        expected_a, expected_b, expected_angle, math.radians(0))
                    expected_y = utils.calculateRadiusAtAngle(
                        expected_a, expected_b, expected_angle, math.radians(90))
                    expected_location_error = math.hypot(
                        expected_x, expected_y) - math.hypot(global_error_x, global_error_y)
                    # cov
                    location_error_std = location_error / expected_location_error
                    # print(location_error, expected_location_error, location_error_std)
                    self.error_tracker_temp.append(
                        [id, location_error_std, length])
                    # print(" error: ", id, location_error, expected_location_error, location_error_std)

                # TruPercept
                trupercept_list = []
                # print(self.localTrackersIDList, self.localTrackersMeasurementList, self.localTrackersExtraList)
                for id_test, mu_test, confidence_test in zip(self.localTrackersIDList, self.localTrackersMeasurementList, self.localTrackersExtraList):
                    # print(self.localTrackersIDList, self.localTrackersMeasurementList, self.localTrackersExtraList)
                    # trupercept_sub_list = []
                    # for id, mu, confidence in zip(self.localTrackersIDList, self.localTrackersMeasurementList, self.localTrackersExtraList):
                    #     if id_test != id:
                    # iou = 1 - shared_math.computeDistanceBBox([mu_test[0], mu_test[1], self.min_size, self.min_size, 0], [mu[0], mu[1], self.min_size, self.min_size, 0])
                    trupercept_list.append([id_test, confidence_test[0]])
                    # trupercept_list.append(trupercept_sub_list)
                self.trupercept_list = trupercept_list

            self.last_update = time
            self.x = self.X_hat_t[0][0]
            self.y = self.X_hat_t[1][0]
            if self.fusion_mode == 2:
                # Different setup for fusion mode 2 need to calculate vector from angle and velocity
                self.dx = self.X_hat_t[2][0] * math.cos(self.X_hat_t[3][0])
                self.dy = self.X_hat_t[2][0] * math.sin(self.X_hat_t[3][0])
            else:
                self.dx = self.X_hat_t[2][0]
                self.dy = self.X_hat_t[3][0]
            self.idx += 1
            if self.P_hat_t[0][0] != 0.0 or self.P_hat_t[0][1] != 0.0:
                self.error_covariance = np.array([[self.P_hat_t[0][0], self.P_hat_t[0][1]], [
                                                 self.P_hat_t[1][0], self.P_hat_t[1][1]]], dtype='float')
                self.d_covariance = np.array([[self.P_hat_t[2][2], self.P_hat_t[2][3]], [
                                             self.P_hat_t[3][2], self.P_hat_t[3][3]]], dtype='float')
            else:
                self.error_covariance = np.array(
                    [[1.0, 0.0], [0.0, 1.0]], dtype='float')
                self.d_covariance = np.array(
                    [[2.0, 0.0], [0.0, 2.0]], dtype='float')

            # except Exception as e:
            #     print ( " Exception: " + str(e) )

    def getKalmanPred(self, time):
        # Prediction based mathcing methods seems to be making this fail so we are using no prediction :/
        # Enforce a min size of a vehicle so that a detection has some area overlap to check
        a, b, phi = utils.ellipsify(self.error_covariance, 1.0)
        return self.x, self.y, a, b, phi


class GlobalTracked:
    def __init__(self, sensor_id, x, y, covariance, dx, dy, dcovariance, confidence, trust_score, time, track_id, fusion_mode, width, length, angle):
        self.x = x
        self.y = y
        self.dx = 0
        self.dy = 0
        self.error_covariance = np.array(
            [[1.0, 0.0], [0.0, 1.0]], dtype='float')
        self.last_measurement = time
        self.last_update = time
        self.id = track_id
        self.idx = 0
        self.min_size = 0.5
        self.track_count = 0
        self.d_covariance = np.array([[2.0, 0.0], [0.0, 2.0]], dtype='float')
        self.match_list = []
        self.fusion_steps = 0
        self.error_monitor = []
        self.num_trackers = 0
        self.width = width
        self.length = length
        self.angle = angle

        # Trupercept stuff
        self.trupercept_list = []

        # Add this first match
        new_match = MatchClass(sensor_id, x, y, covariance, dx,
                               dy, dcovariance, confidence, trust_score, 0, time, width, length, angle)
        self.match_list.append(new_match)

        # Kalman stuff
        self.fusion_mode = fusion_mode
        self.kalman = ResizableKalman(time, x, y, fusion_mode)

    def update(self, other, time):
        new_match = MatchClass(other[0], other[1], other[2], other[3],
                               other[4], other[5], other[6], other[7], other[8], 0, time, other[9], other[10], other[11])
        self.match_list.append(new_match)

        self.last_measurement = time

        self.track_count += 1

    # Gets our position in an array form so we can use it in the BallTree
    def getPosition(self):
        return [
            [self.x, self.y, self.width + 1.0, self.length + 2.0, self.angle]
        ]

    def getPositionPredicted(self, timestamp):
        if self.fusion_steps <= 4:
            # If this kalman filter has never been run, we can't use it for prediction!
            return self.getPosition()
        else:
            # Calculate the time elapsed since the last update
            elapsed_time = timestamp - self.last_update

            # Predict the new position based on the velocity (dx, dy)
            predicted_x = self.x + self.dx * elapsed_time
            predicted_y = self.y + self.dy * elapsed_time

            # Get the uncertainty from the Kalman filter
            _, _, a, b, phi = self.kalman.getKalmanPred(timestamp)

            # Return the predicted position with the current width, length, and angle
            return [
                [predicted_x, predicted_y, self.width + 1.0, self.length + 3.0, self.angle]
            ]

    def fusion(self, time, monitor):
        # self.kalman.fusion(self.match_list, time, monitor)
        # self.x = self.kalman.x
        # self.y = self.kalman.y
        # self.error_covariance = self.kalman.error_covariance
        # self.dx = self.kalman.dx
        # self.dy = self.kalman.dy
        # self.d_covariance = self.kalman.d_covariance
        # self.error_monitor = self.kalman.error_tracker_temp
        # self.num_trackers = len(self.kalman.localTrackersIDList)
        # self.trupercept_list = self.kalman.trupercept_list
        self.fusion_steps += 1

        # Calculate the weighted average width, length, angle, x, and y
        if len(self.match_list) > 0:
            total_weight = 0
            weighted_x = 0
            weighted_y = 0
            weighted_width = 0
            weighted_length = 0
            weighted_angle = 0

            for match in self.match_list:
                weight = 1 / np.linalg.det(match.covariance)
                total_weight += weight
                weighted_x += match.x * weight
                weighted_y += match.y * weight
                weighted_width += match.width * weight
                weighted_length += match.length * weight
                weighted_angle += match.angle * weight

            self.x = weighted_x / total_weight
            self.y = weighted_y / total_weight
            self.width = weighted_width / total_weight
            self.length = weighted_length / total_weight
            self.angle = weighted_angle / total_weight

    def clearLastFrame(self):
        self.match_list = []


class Fusion:
    # Fusion is a special class for matching and fusing detections for a variety of sources.
    # The input is scalable and therefore must be generated before being fed into this class.
    # A unique list of detections is required from each individual sensor or pre-fused device
    # output or it will not be matched. Detections too close to each other may be combined.
    # This is a modified version of the frame-by-frame tracker seen in:
    # https://github.com/eandert/Jetson_Nano_Camera_Vehicle_Tracker
    def __init__(self, id):
        # Set other parameters for the class
        self.tracked_list = []
        self.id = id
        self.current_track_id = 0
        self.min_size = 0.5
        self.trackShowThreshold = 4
        self.fusion_mode = 1

        # Indicate our success
        print(str(self.id) + ' Started FUSION successfully...')

    def fuseDetectionFrame(self, time, monitor=False):
        result = []
        cooperative_monitoring = []
        trupercept_monitoring = []
        detected_objects = []  # New list to store DetectedObject instances

        for track in self.tracked_list:
            track.fusion(time, monitor)
            if track.fusion_steps > self.trackShowThreshold:
                result.append([track.id, track.x, track.y, track.error_covariance.tolist(
                ), track.dx, track.dy, track.d_covariance.tolist(), track.num_trackers])
                if track.error_monitor:
                    cooperative_monitoring.append(track.error_monitor)
                    trupercept_monitoring.append(track.trupercept_list)

                # print("Track ID: ", track.id, "X: ", track.x, "Y: ", track.y, "DX: ", track.dx, "DY: ", track.dy)
                
                # Create a DetectedObject instance and append to the list
                angle = track.angle  # Use the averaged angle
                half_width = track.width / 2
                half_length = track.length / 2

                # Calculate the corners of the bounding box
                cos_angle = math.cos(angle)
                sin_angle = math.sin(angle)

                # Center of the bounding box
                cx, cy = track.x, track.y

                # Calculate the four corners of the bounding box
                bbox = [
                    (cx - half_width * cos_angle - half_length * sin_angle, cy - half_width * sin_angle + half_length * cos_angle),
                    (cx + half_width * cos_angle - half_length * sin_angle, cy + half_width * sin_angle + half_length * cos_angle),
                    (cx + half_width * cos_angle + half_length * sin_angle, cy + half_width * sin_angle - half_length * cos_angle),
                    (cx - half_width * cos_angle + half_length * sin_angle, cy - half_width * sin_angle - half_length * cos_angle)
                ]

                detected_object = DetectedObject(
                    vehicle_id=track.id,
                    vehicle_type=None,  # Assuming track has a type attribute
                    detected_bbox=bbox,  # Bounding box calculated from position, angle, and min size
                    centroid=[track.x, track.y],
                    width=track.width,
                    length=track.length,
                    angle=angle,  # Use the averaged angle
                    expected_error_gaussian=None,
                    error_covariance=track.error_covariance,
                    velocity_vector=[track.dx, track.dy]
                )
                detected_objects.append(detected_object)

            track.clearLastFrame()

        # if len(self.tracked_list) >= 1:
        #     self.cleanDetections()

        return result, detected_objects, cooperative_monitoring, trupercept_monitoring

    def processDetectionFrame(self, timestamp, observations, cleanupTime, trust_score=1.0):
        detections_list_positions = []
        detections_list = []
        for det in observations:
            detections_list_positions.append(
                [det.centroid[0], det.centroid[1], det.dimensions[0], det.dimensions[1], det.angle])
            if det.expected_error_gaussian is not None:
                detections_list.append([det.vehicle_id, det.centroid[0], det.centroid[1], np.array(
                    det.expected_error_gaussian.covariance.tolist()), det.dimensions[0], det.dimensions[1], np.array(det.velocity_vector), det.type, trust_score, det.dimensions[0], det.dimensions[1], det.angle])
            else:
                detections_list.append([det.vehicle_id, det.centroid[0], det.centroid[1], np.array(
                    det.error_covariance), det.dimensions[0], det.dimensions[1], np.array(det.velocity_vector), det.type, trust_score, det.dimensions[0], det.dimensions[1], det.angle])

        # Call the matching function to modify our detections in tracked_list
        self.matchDetections(detections_list_positions, detections_list,
                             timestamp, cleanupTime)

    def matchDetections(self, detections_list_positions, detection_list, timestamp, cleanupTime):
        matches = []
        
        # Check if there are any detections
        if len(detections_list_positions) > 0:
            
            # Check if there are any existing tracks
            if len(self.tracked_list) > 0:
                numpy_formatted = np.array(detections_list_positions).reshape(len(detections_list_positions), 5)
                thisFrameTrackTree = BallTree(numpy_formatted, metric=utils.computeDistanceBBox)

                length = len(numpy_formatted)
                if length > 0:
                    for tracked_listIdx, track in enumerate(self.tracked_list):
                        # Query the BallTree for the nearest neighbors
                        tuple = thisFrameTrackTree.query(np.array(track.getPositionPredicted(timestamp)), k=length, return_distance=True)
                        first = True
                        for IOUVsDetection, detectionIdx in zip(tuple[0][0], tuple[1][0]):
                            if IOUVsDetection < 0.75: # Score is 1 - IOU
                                if first:
                                    try:
                                        index = [i[0] for i in matches].index(detectionIdx)
                                        if matches[index][2] < IOUVsDetection:
                                            matches.append([detectionIdx, tracked_listIdx, IOUVsDetection])
                                            matches[index][2] = 1
                                            matches[index][1] = -99
                                            first = False
                                    except:
                                        matches.append([detectionIdx, tracked_listIdx, IOUVsDetection])
                                        first = False
                                else:
                                    if detectionIdx not in [i[0] for i in matches]:
                                        matches.append([detectionIdx, -99, 1])

                # Update the tracks with the matches
                for match in matches:
                    if match[1] != -99:
                        self.tracked_list[match[1]].relations.append([match[0], match[2]])

                # Update the tracks based on the relations
                for track in self.tracked_list:
                    if len(track.relations) == 1:
                        track.update(detection_list[track.relations[0][0]], timestamp)
                    elif len(track.relations) > 1:
                        max = 0
                        idx = -99
                        for rel in track.relations:
                            if rel[1] < max:
                                max = rel[1]
                                idx = rel[0]

                        if idx != -99:
                            track.update(detection_list[idx], timestamp)

                # Determine which detections were not matched
                if len(matches):
                    missing = sorted(set(range(0, len(detections_list_positions))) - set([i[0] for i in matches]))
                else:
                    missing = list(range(0, len(detections_list_positions)))

                added = []
                for add in missing:
                    added.append(add)
                    new = GlobalTracked(detection_list[add][0], detection_list[add][1], detection_list[add][2],
                                        detection_list[add][3], detection_list[add][4], detection_list[add][5],
                                        detection_list[add][6], detection_list[add][7], detection_list[add][8],
                                        timestamp, (max_id * self.id) + self.current_track_id, self.fusion_mode, detection_list[add][9], detection_list[add][10], detection_list[add][11])
                    if self.current_track_id < max_id:
                        self.current_track_id += 1
                    else:
                        self.current_track_id = 0
                    self.tracked_list.append(new)

            else:
                # If there are no existing tracks, create new tracks for all detections
                for dl in detection_list:
                    new = GlobalTracked(dl[0], dl[1], dl[2], dl[3], dl[4], dl[5],
                                        dl[6], dl[7], dl[8], timestamp, (max_id * self.id) + self.current_track_id, self.fusion_mode, dl[9], dl[10], dl[11])
                    if self.current_track_id < max_id:
                        self.current_track_id += 1
                    else:
                        self.current_track_id = 0
                    self.tracked_list.append(new)

        # Clean up old tracks
        remove = []
        for idx, track in enumerate(self.tracked_list):
            track.relations = []
            if track.last_measurement <= (timestamp - cleanupTime):
                remove.append(idx)

        for delete in reversed(remove):
            self.tracked_list.pop(delete)

    def cleanDetections(self):
        detections_position_list = []
        detections_position_list_id = []
        for track in self.tracked_list:
            a, b, phi = utils.ellipsify(track.error_covariance, 3.0)
            detections_position_list.append(
                [track.x, track.y, track.width, track.length, track.angle])
            detections_position_list_id.append(track.id)
        matches = []
        remove = []
        if len(detections_position_list) > 0:
            if len(self.tracked_list) > 0:
                numpy_formatted = np.array(detections_position_list).reshape(
                    len(detections_position_list), 5)
                thisFrameTrackTree = BallTree(
                    numpy_formatted, metric=utils.computeDistanceBBox)

                # Need to check the tree size here in order to figure out if we can even do this
                length = len(numpy_formatted)
                if length > 0:
                    for tracked_listIdx, track in enumerate(self.tracked_list):
                        # The only difference between this and our other version is that
                        # the below line is commented out
                        # track.calcEstimatedPos(timestamp - self.prev_time)
                        tuple = thisFrameTrackTree.query(np.array(track.getPosition()), k=length,
                                                         return_distance=True)
                        first = True
                        for IOUVsDetection, detectionIdx in zip(tuple[0][0], tuple[1][0]):
                            # 100% match is ourself! Look for IOU > .50 for now to delete
                            if .25 >= IOUVsDetection > 0.001:
                                # Only grab the first match
                                # Before determining if this is a match check if this detection has been matched already
                                if first:
                                    try:
                                        index = [i[0] for i in matches].index(
                                            detectionIdx)
                                        # We have found the detection index, lets see which track is a better match
                                        if matches[index][2] > IOUVsDetection:
                                            # We are better so add ourselves
                                            matches.append(
                                                [detectionIdx, tracked_listIdx, IOUVsDetection])
                                            # Now unmatch the other one because we are better
                                            # This essentiall eliminates double matching
                                            matches[index][2] = 1
                                            matches[index][1] = -99
                                            # Now break the loop
                                            first = False
                                    except:
                                        # No matches in the list, go ahead and add
                                        matches.append(
                                            [detectionIdx, tracked_listIdx, IOUVsDetection])
                                        first = False
                                else:
                                    # The other matches need to be marked so they arent made into a new track
                                    # Set distance to 1 so we know this wasn't the main match
                                    if detectionIdx not in [i[0] for i in matches]:
                                        # No matches in the list, go ahead and add
                                        matches.append([detectionIdx, -99, 1])

                # update the tracks that made it through
                for match in matches:
                    if match[1] != -99:
                        if match[0] != match[1]:
                            if match[1] not in remove and match[0] not in remove:
                                # Check which track is older and keep that one
                                check0 = self.tracked_list[match[0]
                                                           ].last_measurement
                                check1 = self.tracked_list[match[1]
                                                           ].last_measurement
                                # Arbitrary tie break towards earlier in the list
                                if check0 > check1:
                                    if self.tracked_list[match[0]].fusion_steps >= self.trackShowThreshold:
                                        bisect.insort(remove, match[1])
                                elif check0 < check1:
                                    if self.tracked_list[match[1]].fusion_steps >= self.trackShowThreshold:
                                        bisect.insort(remove, match[0])
                                else:
                                    check0 = self.tracked_list[match[0]
                                                               ].fusion_steps
                                    check1 = self.tracked_list[match[1]
                                                               ].fusion_steps
                                    if check0 >= check1:
                                        if check0 >= self.trackShowThreshold:
                                            bisect.insort(remove, match[1])
                                    else:
                                        if check1 >= self.trackShowThreshold:
                                            bisect.insort(remove, match[0])

        for delete in reversed(remove):
            # print("Cleaning track ", delete)
            self.tracked_list.pop(delete)
        # print(len(self.tracked_list), len(remove), remove)