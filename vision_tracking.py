#!/usr/bin/env python

# Author: Brendan Burkhart
# Date: 2022-06-21

# (C) Copyright 2022 Johns Hopkins University (JHU), All Rights Reserved.

# --- begin cisst license - do not edit ---

# This software is provided "as is" under an open source license, with
# no warranty.  The complete license can be found in license.txt and
# http://www.cisst.org/cisst/license.txt.

# --- end cisst license ---

import collections
import cv2
import math
import numpy as np
import threading
import queue
import camera
import time

# Represents a single tracked detection, with location history information
class TrackedObject:
    def __init__(self, detection, max_strength=10, max_history=200, drop_off=2):
        self.position, self.size, self.contour = detection

        # 'signal-strength' of this detection
        self.strength = 1
        self.stale = False
        self.max_strength = max_strength
        self.drop_off = drop_off

        # deque will 'remember' last max_history object locations
        self.location_history = collections.deque(maxlen=max_history)
        self.location_history.append(self.position)

    # l2 distance between this object and 'position'
    def distance_to(self, position):
        dx = self.position[0] - position[0]
        dy = self.position[1] - position[1]
        return math.sqrt(dx * dx + dy * dy)

    def is_strong(self):
        return self.strength >= (self.max_strength - 2)

    def is_stale(self):
        return self.stale

    # when a match is found, known position is updated and strength increased by 1,
    # if no match is found, strength decays by drop_off to prevent strength from oscillating
    def update(self, detection):
        if detection is not None:
            self.position, self.size, self.contour = (
                detection.position,
                detection.size,
                detection.contour,
            )
            self.location_history.append(self.position)

            # cap strength so objects never remain too long
            self.strength = min(self.strength + 1, self.max_strength)
            self.stale = False
        else:
            self.strength = max(self.strength - self.drop_off, 0)
            self.stale = True


# Track all detected objects as they move over time
class ObjectTracker:
    class Parameters:
        def __init__(
            self,
            max_distance=0.02,
            max_history=200,
            max_strength=15,
            drop_off=2,
            unique_target=False,
        ):
            self.max_distance = max_distance
            self.max_history = max_history
            self.max_strength = max_strength
            self.drop_off = drop_off
            self.unique_target = unique_target

    # max_distance is how far objects can move between frames and still be considered
    # the same object. expressed as fraction of maximum image dimension
    def __init__(self, parameters):
        self.objects = []
        self.primary_target = None
        self.parameters = parameters
        self.max_distance_pixels = 1000 * self.parameters.max_distance

    def configure_image_size(self, shape):
        image_size = max(shape)
        self.max_distance_pixels = image_size * self.parameters.max_distance

    # mark on object as the primary object to track
    def set_primary_target(self, position, margin=1.35):
        is_nearby = lambda object: object.distance_to(position) < margin * object.size
        nearby_objects = [x for x in self.objects if is_nearby(x)]

        self.primary_target = nearby_objects[0] if len(nearby_objects) == 1 else None
        if self.primary_target is not None:
            self.primary_target.location_history.clear()

    # Clear location history of all tracked objects
    def clear_history(self):
        for obj in self.objects:
            obj.location_history.clear()

    def _register_unique(self, detections):
        count = len(detections)
        if count == 0:
            if self.primary_target is not None:
                self.primary_target.update(None)
        elif count == 1:
            obj = TrackedObject(detections[0], max_history=self.parameters.max_history)
            if self.primary_target is not None:
                self.primary_target.update(obj)
            else:
                self.primary_target = obj
                self.objects = [self.primary_target]
        else:
            self.primary_target.update(None)
            print("WARNING: unique_target set but found {} targets".format(count))

        if self.primary_target is None or self.primary_target.strength == 0:
            self.primary_target = None
            self.objects = []

    # register detections from the current frame with tracked objects,
    # removing, updating, and adding tracked objects as needed
    def register(self, detections):
        if self.parameters.unique_target:
            self._register_unique(detections)
            return

        # create tracked obejcts for detections, mark all detections as unmatched
        detections = [
            [
                TrackedObject(
                    d,
                    max_strength=self.parameters.max_strength,
                    max_history=self.parameters.max_history,
                    drop_off=self.parameters.drop_off,
                ),
                False,
            ]
            for d in detections
        ]

        # find closest unmatched detection to object
        def closest_detection(obj):
            index, min_distance = None, self.max_distance_pixels

            for i, (detection, matched) in enumerate(detections):
                distance = obj.distance_to(detection.position)
                if not matched and distance < min_distance:
                    index, min_distance = i, distance

            return index

        # update all currently tracked objects
        for obj in self.objects:
            index = closest_detection(obj)
            if index is None:
                obj.update(None)
            else:
                detections[index][1] = True  # mark detection as matched
                obj.update(detections[index][0])

        # detections not matched to existing tracked objects become new tracked objects
        for detection, matched in detections:
            if not matched:
                self.objects.append(detection)

        # remove stale tracked objects and check if primary target is stale
        self.objects = [x for x in self.objects if x.strength > 0]

        if self.primary_target not in self.objects and self.primary_target is not None:
            print("Lost track of target! Please click on target to re-acquire")
            self.primary_target = None


class ColorTarget:
    def __init__(self, min_color_hsv, max_color_hsv, blur_aperature, contour_min_size):
        self.min_color = min_color_hsv
        self.max_color = max_color_hsv
        self.blur_aperature = blur_aperature
        self.contour_min_size = contour_min_size

    def find(self, image):
        def detection(contour):
            M = cv2.moments(contour)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            size = math.sqrt(M["m00"] / math.pi)
            return (center, size, contour)

        # Process a frame - find, track, outline all potential targets
        blurred = cv2.medianBlur(image, 2 * self.blur_aperature + 1)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        thresholded = cv2.inRange(hsv, self.min_color, self.max_color)

        contours, _ = cv2.findContours(
            thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Make contour_min_size relative to maximum image dimension
        contour_min_size = self.contour_min_size * max(image.shape)
        contours = [c for c in contours if cv2.contourArea(c) > contour_min_size]
        # De-nest contour points
        contours = [c.reshape(-1, 2) for c in contours]
        detections = [detection(c) for c in contours]

        return detections


class ArUcoTarget:
    def __init__(self, marker_size, aruco_dict, allowed_ids):
        self.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict)
        self.aruco_parameters = cv2.aruco.DetectorParameters_create()
        self.aruco_parameters.adaptiveThreshWinSizeMin = 10
        self.aruco_parameters.adaptiveThreshWinSizeMax = 40
        self.aruco_parameters.adaptiveThreshWinSizeStep = 10
        # self.aruco_parameters.minMarkerPerimeterRate = 0.005
        # self.aruco_parameters.polygonalApproxAccuracyRate = 0.15
        # self.aruco_parameters.minMarkerDistanceRate = 0.005
        # self.aruco_parameters.minDistanceToBorder = 1
        # self.aruco_parameters.perspectiveRemoveIgnoredMarginPerCell = 0.2
        # self.aruco_parameters.perspectiveRemovePixelPerCell = 10
        # self.aruco_parameters.maxErroneousBitsInBorderRate = 0.6
        # self.aruco_parameters.errorCorrectionRate = 1.0

        self.allowed_ids = allowed_ids
        self.marker_size = marker_size

    def find(self, image):
        def detection(corners):
            center = np.int0(np.mean(corners, axis=0))
            size = math.sqrt(cv2.contourArea(corners))
            return ((center[0], center[1]), size, corners)

        corners, ids, _ = cv2.aruco.detectMarkers(image, self.aruco_dict, parameters=self.aruco_parameters)

        # de-nest data array
        corners = [c[0] for c in corners]
        ids = [x[0] for x in ids] if ids is not None else []

        corners = [corners[i] for i in range(len(ids)) if ids[i] in self.allowed_ids]
        detections = [detection(c) for c in corners]

        return detections


class VisionTracker:
    class Parameters:
        def __init__(self, point_history_length=5):
            self.point_history_length = point_history_length

    def __init__(
        self,
        object_tracker: ObjectTracker,
        target_type,
        camera: camera.Camera,
        parameters=Parameters(),
        window_title="Vision tracking",
    ):
        self.objects = object_tracker
        self.target_type = target_type
        self.parameters = parameters
        self.window_title = window_title
        self.camera = camera
        self.robot_axes = None
        self.points = []
        # move image so callbacks run in correct thread
        self.image_queue = queue.Queue(maxsize=1)

    def _mouse_callback(self, event, x, y, flags, params):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        self.objects.set_primary_target((x, y))

    def _create_window(self):
        cv2.namedWindow(self.window_title)
        cv2.setMouseCallback(self.window_title, self._mouse_callback)

    def _close(self):
        self.camera.set_callback(None)
        cv2.destroyWindow(self.window_title)

    def _process_targets(self, image):
        detections = self.target_type.find(image)
        self.objects.register(detections)

        for obj in self.objects.objects:
            color = (0, 255, 0)
            if self.objects.primary_target == obj:
                color = (255, 0, 255)

            if obj.is_stale():
                color = (0, 255, 255)

            contour = np.array([(x, y) for [x, y] in obj.contour], dtype=np.int)
            cv2.drawContours(image, [contour], -1, color, 3)

    def clear_history(self):
        self.objects.clear_history()

    def set_robot_axes(self, rvec, tvec):
        scale = 0.10
        axes = scale * np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.robot_axes = self.camera.project_points(axes, rvec, tvec)

    def display_points(self, points, rvec, tvec, color):
        projected_points = self.camera.project_points(points, rvec, tvec)
        self.points.append((projected_points, color))

    def display_points_2d(self, points, color):
        self.points.append((points, color))

    def draw_points(self, frame):
        for points, color in self.points:
            for p in points:
                cv2.circle(frame, tuple(p), 2, color)

    def draw_axes(self, frame, axes, color):
        if axes is None:
            return

        for i in range(3):
            start = tuple(np.int0(axes[0]))
            end = tuple(np.int0(axes[i + 1]))
            cv2.line(frame, start, end, color, 3)

    def _add_image(self, image):
        # replace old image if not read yet
        try:
            self.image_queue.get(block=False)
        except queue.Empty:
            pass

        self.image_queue.put(image, block=False)

    # In background, run object tracking and display video
    def start(self, enter_handler, quit_handler):
        self.should_stop = False
        self._enter_handler = enter_handler
        self._quit_handler = quit_handler
        self._should_run_point_acquisition = False
        self._should_run_pose_acquisition = False
        self._should_run_rcm_tracking = False

        self.camera.set_callback(self._add_image)

        def run_camera():
            self._create_window()

            while not self.should_stop:
                try:
                    frame = self.image_queue.get(block=True, timeout=1)
                except queue.Empty:
                    print("No camera image available, waited for 1 second")
                    self._quit_handler()
                    continue

                self._process_targets(frame)

                if self._should_run_point_acquisition:
                    self._run_point_acquisition(frame)

                if self._should_run_pose_acquisition:
                    self._run_target_pose_acquisition(frame)

                self.draw_axes(frame, self.robot_axes, (255, 255, 0))
                self.draw_points(frame)
                cv2.imshow(self.window_title, frame)
                key = cv2.waitKey(20)
                key = key & 0xFF  # Upper bits are modifiers (control, alt, etc.)
                escape = 27
                if key == ord("q") or key == escape:
                    self._quit_handler()
                elif key == ord("d") or key == ord("\n") or key == ord("\r"):
                    self._enter_handler()

            self._close()

        self.background_task = threading.Thread(target=run_camera)
        self.background_task.start()
        return True

    def stop(self):
        self.should_stop = True

    def get_camera_frame(self):
        return self.camera.get_camera_frame()

    def target_visible(self):
        return self.objects.primary_target is not None

    def _run_target_pose_acquisition(self, frame):
        target = self.objects.primary_target
        if target is not None and target.is_strong():
            cv2.circle(
                frame,
                target.position,
                radius=3,
                color=(0, 0, 255),
                thickness=cv2.FILLED,
            )

            # once at least 5 data points have been collected since user selected
            # target, output average location of target
            if len(target.location_history) > self.parameters.point_history_length:
                corners = np.array([target.contour])
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.target_type.marker_size, self.camera.camera_matrix, self.camera.no_distortion)
                cv2.drawFrameAxes(frame, self.camera.camera_matrix, self.camera.no_distortion, rvecs[0], tvecs[0], 0.01)
                rotation, _ = cv2.Rodrigues(rvecs[0])
                self._acquired_pose = (rotation, tvecs[0, 0])

    def _run_point_acquisition(self, frame):
        target = self.objects.primary_target
        if target is not None and target.is_strong():
            cv2.circle(
                frame,
                target.position,
                radius=3,
                color=(0, 0, 255),
                thickness=cv2.FILLED,
            )

            # once at least 5 data points have been collected since user selected
            # target, output average location of target
            if len(target.location_history) > self.parameters.point_history_length:
                mean = np.mean(target.location_history, axis=0)
                self._acquired_point = np.int32(mean)

    # Get location of target
    def acquire_point(self, timeout=None):
        self.objects.clear_history()
        self._acquired_point = None
        self._should_run_point_acquisition = True

        if self.objects.primary_target is None:
            print("Please click target on screen")

        start = time.time()

        while not self.should_stop:
            if timeout is not None and (time.time() - start) > timeout:
                return False, None

            if self._acquired_point is not None:
                return True, self._acquired_point

        return False, None

    # Get pose of target
    def acquire_pose(self, timeout=None):
        self.objects.clear_history()
        self._acquired_pose = None
        self._should_run_pose_acquisition = True

        start = time.time()

        while not self.should_stop:
            if timeout is not None and (time.time() - start) > timeout:
                return False, None

            if self._acquired_pose is not None:
                return True, self._acquired_pose

        return False, None
