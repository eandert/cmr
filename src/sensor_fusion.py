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
        process_variation = .9 # Increased from .7

        # Track the time of the last track
        self.last_measurement = time
        self.last_update = time # Initialize last_update here

        # Track the number of times this Kalman filter has been used
        self.idx = 0

        # The amount of time before a tracker is removed from the list
        self.time_until_removal = .3

        # Store the fusion mode
        self.fusion_mode = fusion_mode

        # Trupercept stuff
        self.trupercept_list = []

        process_noise_estimate = .7 # Increased from .5

        # Set up the Kalman filter
        # Initial State cov
        if self.fusion_mode == 0:
            # Set up the Kalman filter
            self.F_t_len = 4
        elif self.fusion_mode == 1:
            # Setup for x_hat = x + dx + dxdx,  y_hat = y + dy + dydy
            self.F_t_len = 6
        else:
            # model from https://journals.sagepub.com/doi/abs/10.1177/0959651820975523
            self.F_t_len = 5

        # Initialize Kalman state and covariance with placeholders of correct dimensions
        # These will be properly set during the first fusion call (self.idx == 0)
        self.X_hat_t = np.zeros((self.F_t_len, 1), dtype='float')
        self.P_hat_t = np.identity(self.F_t_len)

        # Now apply mode-specific initializations
        if self.fusion_mode == 0:
            # Setup for x_hat = x + dx,  y_hat = y + dy
            # Initial State cov
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
            self.B_t = np.array([[0]] * self.F_t_len, dtype='float') # Adjusted dimensions based on F_t_len
            # Control vector
            self.U_t = 0
        elif self.fusion_mode == 1:
            # Setup for x_hat = x + dx + dxdx,  y_hat = y + dy + dydy
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
            self.B_t = np.array([[0]] * self.F_t_len, dtype='float') # Adjusted dimensions based on F_t_len
            # Control vector
            self.U_t = 0
        else:
            # model from https://journals.sagepub.com/doi/abs/10.1177/0959651820975523
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
            self.B_t = np.array([[0]] * self.F_t_len, dtype='float') # Adjusted dimensions based on F_t_len
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
        # Use the current Kalman state to predict the position at the given time
        elapsed = time - self.last_update
        if elapsed < 0:
            # If the requested time is before the last update, return the last updated position
            # Or handle as an error/special case, for now, just return current state
            predicted_x = self.X_hat_t[0][0]
            predicted_y = self.X_hat_t[1][0]
            predicted_P = self.P_hat_t
        else:
            # Predict the state forward
            if self.fusion_mode == 0:
                F_pred = np.array([[1, 0, elapsed, 0],
                                   [0, 1, 0, elapsed],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], dtype='float')
            elif self.fusion_mode == 1:
                F_pred = np.array([[1, 0, elapsed, 0, elapsed*elapsed, 0],
                                   [0, 1, 0, elapsed, 0, elapsed*elapsed],
                                   [0, 0, 1, 0, elapsed, 0],
                                   [0, 0, 0, 1, 0, elapsed],
                                   [0, 0, 0, 0, 1, 0],
                                   [0, 0, 0, 0, 0, 1]], dtype='float')
            else:
                # For fusion mode 2, the F_t is more complex and depends on current state
                # For simplicity here, we'll use a basic prediction, or if detailed prediction is needed,
                # we might need to re-evaluate the F_t calculation from the fusion method.
                # For now, let's just use the current state as a simple prediction if elapsed > 0 and mode is 2
                # TODO: Implement proper F_pred for mode 2 if needed for matching
                predicted_x = self.X_hat_t[0][0]
                predicted_y = self.X_hat_t[1][0]
                predicted_P = self.P_hat_t
                a, b, phi = utils.ellipsify(predicted_P[0:2, 0:2], 1.0)
                return predicted_x, predicted_y, a, b, phi

            predicted_X_hat, predicted_P = utils.kalman_prediction(
                self.X_hat_t, self.P_hat_t, F_pred, self.B_t, self.U_t, self.Q_t)
            predicted_x = predicted_X_hat[0][0]
            predicted_y = predicted_X_hat[1][0]

        a, b, phi = utils.ellipsify(predicted_P[0:2, 0:2], 1.0)
        return predicted_x, predicted_y, a, b, phi


class GlobalTracked:
    def __init__(self, detected_object, time, track_id, fusion_mode):
        self.x = detected_object.centroid[0]
        self.y = detected_object.centroid[1]
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
        self.width = detected_object.dimensions[0]
        self.length = detected_object.dimensions[1]
        self.angle = detected_object.angle

        # Trupercept stuff
        self.trupercept_list = []

        # Add this first match
        new_match = MatchClass(detected_object.vehicle_id, detected_object.centroid[0], detected_object.centroid[1], detected_object.error_covariance,
                               detected_object.velocity_vector[0], detected_object.velocity_vector[1], detected_object.error_covariance, 1.0, 1.0, 0, time, detected_object.dimensions[0], detected_object.dimensions[1], detected_object.angle)
        self.match_list.append(new_match)

        # Kalman stuff
        self.fusion_mode = fusion_mode
        self.kalman = ResizableKalman(time, self.x, self.y, fusion_mode)

    def update(self, detected_object, time):
        new_match = MatchClass(detected_object.vehicle_id, detected_object.centroid[0], detected_object.centroid[1], detected_object.error_covariance,
                               detected_object.velocity_vector[0], detected_object.velocity_vector[1], detected_object.error_covariance, 1.0, 1.0, 0, time, detected_object.dimensions[0], detected_object.dimensions[1], detected_object.angle)
        self.match_list.append(new_match)

        self.last_measurement = time

    # Gets our position in an array form so we can use it in the BallTree
    def getPosition(self):
        return [
            [self.x, self.y, self.width, self.length, self.angle]
        ]

    def getPositionPredicted(self, timestamp):
        # Call the Kalman filter's prediction method
        predicted_x, predicted_y, a, b, phi = self.kalman.getKalmanPred(timestamp)

        # Dynamically adjust the bounding box size based on Kalman filter's uncertainty
        # Use 3 standard deviations for the matching gate
        adjusted_width = self.width + (a * 3.0) # Multiply 'a' (semi-major axis of error ellipse) by a factor
        adjusted_length = self.length + (b * 3.0) # Multiply 'b' (semi-minor axis of error ellipse) by a factor

        # Return the predicted position with the dynamically adjusted width and length
        return [
            [predicted_x, predicted_y, adjusted_width, adjusted_length, self.angle]
        ]

    def fusion(self, time, monitor):
        self.kalman.fusion(self.match_list, time, monitor)
        self.x = self.kalman.x
        self.y = self.kalman.y
        self.error_covariance = self.kalman.error_covariance
        self.dx = self.kalman.dx
        self.dy = self.kalman.dy
        self.d_covariance = self.kalman.d_covariance
        self.error_monitor = self.kalman.error_tracker_temp
        self.num_trackers = len(self.kalman.localTrackersIDList)
        self.trupercept_list = self.kalman.trupercept_list
        self.fusion_steps += 1

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
                detected_object = DetectedObject(
                    vehicle_id=track.id,
                    vehicle_type=None,  # Assuming track has a type attribute
                    detected_bbox=None,  # Bounding box will be calculated by the property
                    centroid=[track.x, track.y],
                    width=track.width,
                    length=track.length,
                    angle=track.angle,  # Use the averaged angle
                    expected_error_gaussian=None,
                    error_covariance=track.error_covariance,
                    velocity_vector=[track.dx, track.dy]
                )
                detected_objects.append(detected_object)

                # Now use the property to get the bounding box corners
                bbox = detected_object.detected_bbox_corners

                # If you need to use the bbox directly later, you can update the detected_bbox attribute
                detected_object.detected_bbox = bbox

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
            detections_list.append(det)

        # Call the matching function to modify our detections in tracked_list
        self.matchDetections(detections_list_positions, detections_list,
                             timestamp, cleanupTime)

    def matchDetections(self, detections_list_positions, detection_list, timestamp, cleanupTime):
        matches = []
        
        # Check if there are any detections
        if len(detections_list_positions) > 0:
            
            # Check if there are any existing tracks
            if len(self.tracked_list) > 0:
                # For small numbers of detections, linear search might be faster than building a tree
                detection_count = len(detections_list_positions)
                
                # Only build BallTree if we have enough detections to justify it
                # BallTree construction is O(n log n), so for small n, linear search is faster
                if detection_count > 10:  # Threshold: use tree for >10 detections
                    numpy_formatted = np.array(detections_list_positions).reshape(len(detections_list_positions), 5)
                    thisFrameTrackTree = BallTree(numpy_formatted, metric=utils.computeDistanceBBox)
                else:
                    # For small numbers, use None to trigger linear search fallback
                    thisFrameTrackTree = None
                    numpy_formatted = np.array(detections_list_positions).reshape(len(detections_list_positions), 5)

                length = len(numpy_formatted)
                if length > 0:
                    for tracked_listIdx, track in enumerate(self.tracked_list):
                        # Query the BallTree for the nearest neighbors (or use linear search for small counts)
                        if thisFrameTrackTree is not None:
                            # Use BallTree for efficient querying
                            tuple = thisFrameTrackTree.query(np.array(track.getPositionPredicted(timestamp)), k=length, return_distance=True)
                            distances = tuple[0][0]
                            indices = tuple[1][0]
                        else:
                            # Linear search for small detection counts (faster than building tree)
                            track_pos_list = track.getPositionPredicted(timestamp)
                            track_pos = track_pos_list[0]  # getPositionPredicted returns [[x, y, w, h, angle]]
                            distances_and_indices = []
                            for idx, det_pos in enumerate(detections_list_positions):
                                # Compute distance using the same metric as BallTree
                                distance = utils.computeDistanceBBox(
                                    track_pos,  # [x, y, width, length, angle]
                                    det_pos
                                )
                                distances_and_indices.append((distance, idx))
                            # Sort by distance and take top k
                            distances_and_indices.sort(key=lambda x: x[0])
                            distances = np.array([d[0] for d in distances_and_indices[:length]])
                            indices = np.array([d[1] for d in distances_and_indices[:length]])
                        
                        first = True
                        for IOUVsDetection, detectionIdx in zip(distances, indices):
                            if IOUVsDetection < 0.5: # Score is 1 - IOU
                                if first:
                                    try:
                                        index = [i[0] for i in matches].index(detectionIdx)
                                        if matches[index][2] > IOUVsDetection:
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
                        max = 100 # Initialize with a high value for distance (1 - IOU)
                        idx = -99
                        for rel in track.relations:
                            if rel[1] < max: # Look for the smallest distance (highest IOU)
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
                    new = GlobalTracked(detection_list[add],
                                        timestamp, (max_id * self.id) + self.current_track_id, self.fusion_mode)
                    if self.current_track_id < max_id:
                        self.current_track_id += 1
                    else:
                        self.current_track_id = 0
                    self.tracked_list.append(new)

            else:
                # If there are no existing tracks, create new tracks for all detections
                for dl in detection_list:
                    new = GlobalTracked(dl, timestamp, (max_id * self.id) + self.current_track_id, self.fusion_mode)
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
                # Optimization #3: Only build BallTree if we have enough detections
                detection_count = len(detections_position_list)
                if detection_count > 10:  # Threshold: use tree for >10 detections
                    thisFrameTrackTree = BallTree(
                        numpy_formatted, metric=utils.computeDistanceBBox)
                else:
                    thisFrameTrackTree = None

                # Need to check the tree size here in order to figure out if we can even do this
                length = len(numpy_formatted)
                if length > 0:
                    for tracked_listIdx, track in enumerate(self.tracked_list):
                        # The only difference between this and our other version is that
                        # the below line is commented out
                        # track.calcEstimatedPos(timestamp - self.prev_time)
                        if thisFrameTrackTree is not None:
                            tuple = thisFrameTrackTree.query(np.array(track.getPosition()), k=length,
                                                             return_distance=True)
                            distances = tuple[0][0]
                            indices = tuple[1][0]
                        else:
                            # Linear search for small detection counts
                            track_pos_list = track.getPosition()
                            track_pos = track_pos_list[0]  # getPosition returns [[x, y, w, h, angle]]
                            distances_and_indices = []
                            for idx, det_pos in enumerate(detections_position_list):
                                distance = utils.computeDistanceBBox(track_pos, det_pos)
                                distances_and_indices.append((distance, idx))
                            distances_and_indices.sort(key=lambda x: x[0])
                            distances = np.array([d[0] for d in distances_and_indices[:length]])
                            indices = np.array([d[1] for d in distances_and_indices[:length]])
                        
                        first = True
                        for IOUVsDetection, detectionIdx in zip(distances, indices):
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