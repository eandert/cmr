import math
import numpy as np
from numpy.testing._private.utils import measure
from sklearn.neighbors import BallTree
import utils

CAMERA = 0
LIDAR = 1
MAX_ID = 10000


class MatchClass:
    def __init__(self, x, y, covariance, dx, dy, d_confidence, sensor_type, confidence, time, sensor_id):
        self.x = x
        self.y = y
        self.covariance = np.array(covariance)
        self.dx = dx
        self.dy = dy
        self.velocity_confidence = d_confidence
        self.sensor_type = sensor_type
        self.last_tracked = time
        self.sensor_id = sensor_id
        self.confidence = confidence


class Tracked:
    # This object tracks a single object that has been detected in a video frame.
    # We use this primarily to match objects seen between frames and included in here
    # is a function for kalman filter to smooth the x and y values as well as a
    # function for prediction where the next bounding box will be based on prior movement.
    def __init__(self, sensor_id, x, y, cov, sensor_type, confidence, time, id):
        self.x = x
        self.y = y
        self.dx = 0
        self.dy = 0
        self.error_covariance = np.array(
            [[1.0, 0.0], [0.0, 1.0]], dtype='float')
        self.last_tracked = time
        self.id = id
        self.idx = 0
        self.min_size = 0.5
        self.dt = .125
        self.track_count = 0
        self.fusion_steps = 0
        self.d_covariance = np.array([[2.0, 0.0], [0.0, 2.0]], dtype='float')
        self.first = True

        self.localTrackersMeasurementList = []
        self.localTrackersCovarianceList = []
        self.localTrackersIDList = []

        # Build the match list and add our match to it
        self.match_list = []
        new_match = MatchClass(x, y, cov, None, None,
                               None, sensor_type, confidence, time, sensor_id)
        self.match_list.append(new_match)

        # Kalman stuff
        process_variation = 0.16

        # Set other parameters for the class
        self.prev_time = -99

        # Set up the Kalman filter
        # Initial State cov
        self.P_hat_t = np.identity(4)
        self.P_hat_t[2][2] = 0.0
        self.P_hat_t[3][3] = 0.0
        # Process cov
        four = process_variation * (.125*.125*.125*.125)/4.0
        three = process_variation * (.125*.125*.125)/3.0
        two = process_variation * (.125*.125)/2.0
        one = process_variation * (.125)
        self.Q_t = np.array([[three, 0, two, 0],
                            [0, three, 0, two],
                            [two, 0, process_variation, 0],
                            [0, two, 0, process_variation]], dtype='float')
        # Control matrix
        self.B_t = np.array([[0], [0], [0], [0]], dtype='float')
        # self.B_t = G
        # Control vector
        # self.U_t = 0
        self.U_t = 0
        # Measurment Matrix
        # Generated on the fly
        # Measurment cov
        self.R_t = np.identity(4)

    # Update adds another detection to this track
    def update(self, other, time):

        new_match = MatchClass(
            other[1], other[2], other[3], None, None, None, other[4], other[5], time, other[0])
        self.match_list.append(new_match)

        self.last_tracked = time

        self.track_count += 1

    # Gets our position in an array form so we can use it in the BallTree
    def getPosition(self):
        return [
            [self.x, self.y, self.min_size, self.min_size, math.radians(0)]
        ]

    def getPositionPredicted(self, timestamp, estimate_covariance):
        if self.fusion_steps < 2 or not estimate_covariance:
            # If this kalman fitler has never been run, we can't use it for prediction!
            return [
                [self.x, self.y, self.min_size, self.min_size, math.radians(0)]
            ]
        else:
            try:
                x, y, a, b, phi = self.getKalmanPred(timestamp)
                return [
                    [x, y, a, b, phi]
                ]
            except:
                return [
                    [self.x, self.y, self.min_size,
                        self.min_size, math.radians(0)]
                ]

    def clearLastFrame(self):
        self.match_list = []

    def fx(self, x, dt):
        """ state transition function for a 
        constant velocity aircraft"""

        F = np.array([[1, 0, dt, 0],
                     [0, 1, 0, dt],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]], dtype='float')

        return F @ x

    def hx(self, x):
        ret_val = self.tempH_t
        ret_val[0][0] = ret_val[0][0] * x[0]
        ret_val[1][1] = ret_val[1][1] * x[1]
        ret_val[2][2] = ret_val[2][0] * x[2]
        ret_val[3][3] = ret_val[3][1] * x[3]
        return ret_val

    def averageMeasurementsFirstFrame(self):
        # This try catch somehow prevents a startup error maybe the len function doesn't work sometimes?
        try:
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
        except:
            return self.localTrackersMeasurementList[0], self.localTrackersCovarianceList[0]

    def fusion(self, predictive):
        self.localTrackersMeasurementList = []
        self.localTrackersCovarianceList = []
        self.localTrackersIDList = []
        self.localTrackersConfidenceList = []

        # Time to go through the track list and fuse!
        for match in self.match_list:
            self.localTrackersCovarianceList.append(match.covariance)
            self.localTrackersMeasurementList.append(
                np.array([match.x, match.y]))
            self.localTrackersIDList.append(match.sensor_type)
            self.localTrackersConfidenceList.append(match.confidence)

        # Now do the kalman thing!
        if self.idx == 0:
            if len(self.match_list) == 0:
                pass
            # We have no prior detection so we need to just output what we have but store for later
            # Do a Naive average to get the starting position
            # print(self.localTrackersMeasurementList, self.localTrackersCovarianceList)
            pos, self.error_covariance = self.averageMeasurementsFirstFrame()
            x_out = pos[0]
            y_out = pos[1]

            # Store so that next fusion is better
            self.X_hat_t = np.array(
                [[x_out], [y_out], [0], [0]], dtype='float')

            # Seed the covariance values directly from the measurement
            try:
                self.P_hat_t[0][0] = self.error_covariance[0][0]
                self.P_hat_t[0][1] = self.error_covariance[0][1]
                self.P_hat_t[1][0] = self.error_covariance[1][0]
                self.P_hat_t[1][1] = self.error_covariance[1][1]
            except:
                self.P_hat_t[0][0] = 1.0
                self.P_hat_t[0][1] = 0.0
                self.P_hat_t[1][0] = 0.0
                self.P_hat_t[1][1] = 1.0
            self.prev_time = self.last_tracked
            self.x = x_out
            self.y = y_out
            self.dx = 0.0
            self.dy = 0.0
            self.idx = 1
        else:
            # Prediction step!
            elapsed = self.last_tracked - self.prev_time
            if elapsed <= 0.0:
                # Set to arbitrary time
                elapsed = 0.125

                self.F_t = np.array([[1, 0, elapsed, 0],
                                    [0, 1, 0, elapsed],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], dtype='float')

            # Time to run our predictions!
            self.X_hat_t, self.P_hat_t = utils.kalman_prediction(
                self.X_hat_t, self.P_hat_t, self.F_t, self.B_t, self.U_t, self.Q_t)

            if len(self.localTrackersMeasurementList) == 0:
                nothing_cov = np.array([[1.0, 0.],
                                        [0., 1.0]], dtype='float')
                measure = np.array([.0, .0], dtype='float')
                nothing_Ht = np.array([[0, 0., 0., 0.],
                                        [0., 0, 0., 0.]], dtype='float')

                Z_t = (measure).transpose()
                Z_t = Z_t.reshape(Z_t.shape[0], -1)
                self.X_hat_t, self.P_hat_t = utils.kalman_update(
                    self.X_hat_t, self.P_hat_t, Z_t, nothing_cov, nothing_Ht)
            else:
                for mu, cov in zip(self.localTrackersMeasurementList, self.localTrackersCovarianceList):
                    h_t = np.array([[1., 0., 0., 0.],
                                    [0., 1., 0., 0.]], dtype='float')
                    Z_t = (mu).transpose()
                    Z_t = Z_t.reshape(Z_t.shape[0], -1)
                    self.X_hat_t, self.P_hat_t = utils.kalman_update(
                        self.X_hat_t, self.P_hat_t, Z_t, cov, h_t)

            # Here we run an extra predictive step since it takes 125 ms to compute our data, but
            # we do not save any of this and will re-run next time
            if predictive:
                self.X_hat_t, self.P_hat_t = utils.kalman_prediction(
                    self.X_hat_t, self.P_hat_t, self.F_t, self.B_t, self.U_t, self.Q_t)

            self.prev_time = self.last_tracked

            self.x = self.X_hat_t[0][0]
            self.y = self.X_hat_t[1][0]
            self.dx = self.X_hat_t[2][0]
            self.dy = self.X_hat_t[3][0]
            self.idx += 1
            if self.P_hat_t[0][0] != 0.0 or self.P_hat_t[0][1] != 0.0:
                self.error_covariance = np.array([[self.P_hat_t[0][0], self.P_hat_t[0][1]], [
                                                 self.P_hat_t[1][0], self.P_hat_t[1][1]]], dtype='float')
                self.d_covariance = np.array([[self.P_hat_t[2][2], self.P_hat_t[2][3]], [
                                             self.P_hat_t[3][2], self.P_hat_t[3][3]]], dtype='float')
            else:
                # print ( " what the heck: ", P_hat_t)
                self.error_covariance = np.array(
                    [[1.0, 0.0], [0.0, 1.0]], dtype='float')
                self.d_covariance = np.array(
                    [[2.0, 0.0], [0.0, 2.0]], dtype='float')

            self.fusion_steps += 1

    def getKalmanPred(self, time):
        # Prediction based matching methods seems to be making this fail so we are using no prediction :/
        # Enforce a min size of a vehicle so that a detection has some area overlap to check
        a, b, phi = utils.ellipsify(self.error_covariance, 1.0)
        return self.x, self.y, self.min_size + a, self.min_size + b, phi


class FUSION:
    def __init__(self, cav_cis_id):
        self.tracked_list = []
        self.id = cav_cis_id
        self.prev_time = -99.0
        self.current_tracked_id = 0
        self.trackShowThreshold = 4
        self.predictive = False

        print('Started FUSION successfully...')

    def fuseDetectionFrame(self):
        result = []
        second_list = []
        for track in self.tracked_list:
            track.fusion(self.predictive)
            if track.fusion_steps >= self.trackShowThreshold:
                universal_id = self.id * MAX_ID + track.id
                result.append([universal_id, track.x, track.y, track.error_covariance.tolist(), track.dx, track.dy, track.d_covariance.tolist()])
                second_list.append({
                    "sensor_type_id": sensor_type.id,
                    "detector_type_id": detector_type.id,
                    "max_range": sensor_type.max_range,
                    "horizontal_fov": sensor_type.horizontal_fov,
                    "center_angle": sensor_type.center_angle,
                    "centroid_radial_error_polynomial": utils.Polynomial(detector_type.centroid_radial_error_polynomial),
                    "centroid_distance_error_polynomial": utils.Polynomial(detector_type.centroid_distance_error_polynomial),
                    "bounding_box_error_polynomial": utils.Polynomial(detector_type.bounding_box_error_polynomial),
                    "detection_probability_polynomial": utils.Polynomial(detector_type.detection_probability_polynomial)
                })
            track.clearLastFrame()
        return result, second_list

    def processDetectionFrame(self, timestamp, observations, cleanupTime, estimate_covariance):
        detections_position_list = []
        detections_list = []
        for det in observations:
            detections_position_list.append(
                [det.centroid[0], det.centroid[1], det.dimensions[0], det.dimensions[1], det.angle])
            detections_list.append(
                    [det.vehicle_id, det.centroid[0], det.centroid[1], det.angle, 0, det.expected_error_gaussian])

        self.matchDetections(detections_position_list, detections_list, timestamp, cleanupTime, estimate_covariance)

    def matchDetections(self, detections_list_positions, detection_list, timestamp, cleanupTime, estimate_covariance):
        matches = []
        if len(detections_list_positions) > 0:
            if len(self.tracked_list) > 0:
                numpy_formatted = np.array(detections_list_positions).reshape(len(detections_list_positions), 5)
                thisFrameTrackTree = BallTree(numpy_formatted, metric=computeDistanceEuclidean)

                length = len(numpy_formatted)
                if length > 0:
                    for tracked_listIdx, track in enumerate(self.tracked_list):
                        tuple = thisFrameTrackTree.query(np.array(track.getPositionPredicted(timestamp, estimate_covariance)), k=length, return_distance=True)
                        first = True
                        for IOUVsDetection, detectionIdx in zip(tuple[0][0], tuple[1][0]):
                            if .99 >= IOUVsDetection >= 0:
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

                for match in matches:
                    if match[1] != -99:
                        self.tracked_list[match[1]].relations.append([match[0], match[2]])

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

                if len(matches):
                    missing = sorted(set(range(0, len(detections_list_positions))) - set([i[0] for i in matches]))
                else:
                    missing = list(range(0, len(detections_list_positions)))

                added = []
                for add in missing:
                    tuple = thisFrameTrackTree.query((np.array([detections_list_positions[add]])), k=length, return_distance=True)
                    add_this = False
                    for IOUsDetection, detectionIdx in zip(tuple[0][0], tuple[1][0]):
                        if .99 <= IOUsDetection:
                            if add != detectionIdx:
                                if detectionIdx not in added:
                                    add_this = True
                                    break
                    if add_this:
                        added.append(add)
                        new = Tracked(detection_list[add][0], detection_list[add][1], detection_list[add][2], detection_list[add][3], detection_list[add][4], detection_list[add][5], timestamp, self.current_tracked_id)
                        if self.current_tracked_id < MAX_ID:
                            self.current_tracked_id += 1
                        else:
                            self.current_tracked_id = 0
                        self.tracked_list.append(new)
            else:
                for dl in detection_list:
                    new = Tracked(dl[0], dl[1], dl[2], dl[3], dl[4], dl[5], timestamp, self.current_tracked_id)
                    if self.current_tracked_id < MAX_ID:
                        self.current_tracked_id += 1
                    else:
                        self.current_tracked_id = 0
                    self.tracked_list.append(new)

        remove = []
        for idx, track in enumerate(self.tracked_list):
            track.relations = []
            if track.last_tracked < (timestamp - cleanupTime):
                remove.append(idx)

        for delete in reversed(remove):
            self.tracked_list.pop(delete)

# This function turns elipses into rectanges so that an IO calculation can be done for 
# ball tree matching
def computeDistanceEllipseBox(a, b):
    cx = a[0]
    cy = a[1]
    w = a[2]
    h = a[3]
    angle = a[4]
    c = box(-w/2.0, -h/2.0, w/2.0, h/2.0)
    rc = rotate(c, angle)
    contour_a = translate(rc, cx, cy)

    cx = b[0]
    cy = b[1]
    w = b[2]
    h = b[3]
    angle = a[4]
    c = box(-w/2.0, -h/2.0, w/2.0, h/2.0)
    rc = rotate(c, angle)
    contour_b = translate(rc, cx, cy)

    iou = contour_a.intersection(contour_b).area / contour_a.union(contour_b).area

    # Modify to invert the IOU so that it works with the BallTree class
    if iou <= 0:
        distance = 1
    else:
        distance = 1 - iou

    return distance