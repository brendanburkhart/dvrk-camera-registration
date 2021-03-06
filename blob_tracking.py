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


# Represents a single tracked detection, with location history information
class TrackedObject:
    def __init__(self, detection, max_strength=10, max_history=200, drop_off=2):
        self.position, self.size, self.contour = detection

        # 'signal-strength' of this detection
        self.strength = 1
        self.stale = False
        self.max_strength = max_strength
        self.drop_off = drop_off

        # queue will 'remember' last max_history object locations
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
    # if no match is found, strength decays by 2 to prevent strength from oscillating
    def update(self, detection):
        if detection is not None:
            self.position, self.size, self.contour = detection
            self.location_history.append(self.position)

            # cap strength so objects never remain too long
            self.strength = min(self.strength + 1, self.max_strength)
            self.stale = False
        else:
            self.strength -= 2
            self.stale = True


# Track all detected objects as they move over time
class ObjectTracking:
    # max_distance is how far objects can move between frames
    # and still be considered the same object
    def __init__(self, max_distance, max_history):
        self.objects = []
        self.primary_target = None
        self.max_distance = max_distance
        self.max_history = max_history

    # mark on object as the primary object to track
    def set_primary_target(self, position):
        is_nearby = lambda object: object.distance_to(position) < 1.35 * object.size
        nearby_objects = [x for x in self.objects if is_nearby(x)]

        self.primary_target = nearby_objects[0] if len(nearby_objects) == 1 else None
        if self.primary_target is not None:
            self.primary_target.location_history.clear()

    # Clear location history of all tracked objects
    def clear_history(self):
        for obj in self.objects:
            obj.location_history.clear()

    # register detections from the current frame with tracked objects,
    # removing, updating, and adding tracked objects as needed
    def register(self, detections):
        # No existing tracked objects, add all detections as new objects
        if len(self.objects) == 0:
            for d in detections:
                self.objects.append(TrackedObject(d, self.max_history))

            return

        # No detections were found
        if len(detections) == 0:
            for obj in self.objects:
                obj.update(None)

            return

        # Matrix of minimum distance between each detection and each tracked object
        distances = np.array(
            [[obj.distance_to(d[0]) for d in detections] for obj in self.objects]
        )

        # Array of closest tracked object to each detection
        closest = np.argmin(distances, axis=0)
        current_object_count = len(self.objects)

        # associate detections with existing tracked objects or add as new objects
        for i, detection in enumerate(detections):
            if distances[closest[i], i] <= self.max_distance:
                self.objects[closest[i]].update(detection)
            else:
                self.objects.append(TrackedObject(detection, self.max_history))

        # Update tracked objects not associated with current detection
        closest = np.argmin(distances, axis=1)
        for j in range(current_object_count):
            if distances[j, closest[j]] > self.max_distance:
                self.objects[j].update(None)

        # Remove stale tracked objects and check if primary target is stale
        self.objects = [x for x in self.objects if x.strength > 0]
        if not self.primary_target in self.objects and self.primary_target is not None:
            print("Lost track of target! Please click on target to re-acquire")
            self.primary_target = None


class ColorTarget:
    def __init__(self, min_color_hsv, max_color_hsv, blur_aperature, contour_min_size):
        self.min_color = min_color_hsv
        self.max_color = max_color_hsv
        self.blur_aperature = blur_aperature
        self.contour_min_size = contour_min_size

    def find(self, image):
        # Process a frame - find, track, outline all potential targets
        blurred = cv2.medianBlur(image, 2 * self.blur_aperature + 1)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        thresholded = cv2.inRange(hsv, self.min_color, self.max_color)

        contours, _ = cv2.findContours(
            thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = [c for c in contours if cv2.contourArea(c) > self.contour_min_size]

        moments = [cv2.moments(c) for c in contours]
        radius = lambda area: math.sqrt(area / math.pi)
        detections = [
            ((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])), radius(M["m00"]), c)
            for c, M in zip(contours, moments)
        ]

        return detections


class ArUcoTarget:
    def __init__(self, aruco_dict, allowed_ids):
        self.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict)
        self.aruco_parameters = cv2.aruco.DetectorParameters_create()
        self.allowed_ids = allowed_ids

    def find(self, image):
        def detection(corners):
            center = np.int0(np.mean(corners, axis=0))
            size = math.sqrt(cv2.contourArea(corners))
            return ((center[0], center[1]), size, corners)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_parameters
        )

        # de-nest data array
        corners = [c[0] for c in corners]
        ids = [x[0] for x in ids] if ids is not None else []

        corners = [corners[i] for i in range(len(ids)) if ids[i] in self.allowed_ids]
        detections = [detection(c) for c in corners]
        
        return detections


class BlobTracker:
    class Parameters:
        def __init__(
            self, point_history_length=5
        ):
            self.point_history_length = point_history_length

    def __init__(
        self,
        object_tracker,
        target_type,
        parameters=Parameters(),
        camera_calibration=None,
        window_title="CV Calibration",
    ):
        self.objects = object_tracker
        self.target_type = target_type
        self.parameters = parameters
        self.window_title = window_title
        self.camera_calibration = camera_calibration
        self.robot_axes = None
        self.points = []

    def _mouse_callback(self, event, x, y, flags, params):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        self.objects.set_primary_target((x, y))

    def _create_window(self):
        cv2.namedWindow(self.window_title)
        cv2.setMouseCallback(self.window_title, self._mouse_callback)

    def _init_video(self):
        self.video_capture = cv2.VideoCapture(0)
        self.video_capture.set(cv2.CAP_PROP_FPS, 30)
        self.video_capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        ok = False
        if self.video_capture.isOpened():
            ok, frame = self.video_capture.read()

        if not ok:
            print("\n\nFailed to read from camera.")
            return False

        if self.camera_calibration is not None:
            self.camera_calibration.configure_image_size(frame.shape)

        return ok

    def __del__(self):
        self.video_capture.release()
        cv2.destroyWindow(self.window_title)

    def _process_targets(self, image):
        detections = self.target_type.find(image)
        self.objects.register(detections)

        for o in self.objects.objects:
            color = (0, 255, 0)
            if self.objects.primary_target == o:
                color = (255, 0, 255)
            
            if o.is_stale():
                color = (0, 255, 255)

            contour = np.array([(x, y) for [x, y] in o.contour], dtype=np.int)
            cv2.drawContours(image, [contour], -1, color, 3)

    def clear_history(self):
        self.objects.clear_history()

    def set_robot_axes(self, rvec, tvec):
        scale = 0.10
        axes = scale * np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.robot_axes = self.camera_calibration.project_points(axes, rvec, tvec)

    def display_points(self, points, rvec, tvec, color):
        projected_points = self.camera_calibration.project_points(points, rvec, tvec)
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

    # In background, run object tracking and display video
    def start(self, enter_handler, quit_handler):
        self.should_stop = False
        self._enter_handler = enter_handler
        self._quit_handler = quit_handler
        self._should_run_point_acquisition = False
        self._should_run_rcm_tracking = False

        def run_camera():
            self._create_window()
            ok = self._init_video()

            while ok and not self.should_stop:
                ok, frame = self.video_capture.read()
                frame = (
                    self.camera_calibration.undistort(frame)
                    if self.camera_calibration is not None
                    else frame
                )
                if not ok:
                    print("\n\nFailed to read from camera")
                    self._quit_handler()

                self._process_targets(frame)

                if self._should_run_point_acquisition:
                    self._run_point_acquisition(frame)

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

        self.background_task = threading.Thread(target=run_camera)
        self.background_task.start()
        return True

    def stop(self):
        self.should_stop = True

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
    def acquire_point(self):
        self.objects.clear_history()
        self._acquired_point = None
        self._should_run_point_acquisition = True

        if self.objects.primary_target is None:
            print("Please click target on screen")

        while not self.should_stop:
            if self._acquired_point is not None:
                return True, self._acquired_point

        return False, None
