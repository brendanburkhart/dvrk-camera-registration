#!/usr/bin/env python

# Author: Brendan Burkhart
# Date: 2022-06-21

# (C) Copyright 2022 Johns Hopkins University (JHU), All Rights Reserved.

# --- begin cisst license - do not edit ---

# This software is provided "as is" under an open source license, with
# no warranty.  The complete license can be found in license.txt and
# http://www.cisst.org/cisst/license.txt.

# --- end cisst license ---

import cv2
import math
import numpy as np
import threading
import queue
import scipy
import camera
import time

class ArUcoTarget:
    def __init__(self, marker_size, aruco_dict, allowed_ids):
        self.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict)
        self.aruco_parameters = cv2.aruco.DetectorParameters_create()
        self.aruco_parameters.adaptiveThreshWinSizeMin = 10
        self.aruco_parameters.adaptiveThreshWinSizeMax = 120
        self.aruco_parameters.adaptiveThreshWinSizeStep = 10
        self.aruco_parameters.cornerRefinementWinSize = 15
        self.aruco_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        self.allowed_ids = allowed_ids
        self.marker_size = marker_size

        self.refinement_window_interp = scipy.interpolate.interp1d([0.0, 0.01, 0.025, 0.03, 0.06], [0.0, 15.0, 23.0, 27.0, 40.0], kind="linear", fill_value="extrapolate")

    def _detect(self, image, parameters):
        corners, ids, _ = cv2.aruco.detectMarkers(image, self.aruco_dict, parameters=parameters)

        ids = [x[0] for x in ids] if ids is not None else []
        corners = [corners[i][0] for i in range(len(ids)) if ids[i] in self.allowed_ids]

        return np.array(corners[0]) if len(corners) == 1 else None

    def _refine_detection(self, image, target):
        image_size = image.shape[0]*image.shape[1]
        relative_contour_size = cv2.contourArea(target)/image_size

        corner_refinement_window_size = self.refinement_window_interp(relative_contour_size)
        corner_refinement_window_size = int(corner_refinement_window_size)
        # print(relative_contour_size, corner_refinement_window_size)

        refined_parameters = cv2.aruco.DetectorParameters_create()
        refined_parameters.adaptiveThreshWinSizeMin = self.aruco_parameters.adaptiveThreshWinSizeMin
        refined_parameters.adaptiveThreshWinSizeMax = self.aruco_parameters.adaptiveThreshWinSizeMax
        refined_parameters.adaptiveThreshWinSizeStep = self.aruco_parameters.adaptiveThreshWinSizeStep
        refined_parameters.cornerRefinementWinSize = corner_refinement_window_size
        refined_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        return self._detect(image, refined_parameters)

    def find(self, image):
        target = self._detect(image, self.aruco_parameters)

        if target is not None:
            return self._refine_detection(image, target)
        else:
            return None

class VisionTracker:
    class Parameters:
        def __init__(self, pose_samples=5):
            self.pose_samples = pose_samples

    def __init__(
        self,
        target_type,
        camera: camera.Camera,
        parameters=Parameters(),
        window_title="Vision tracking",
    ):
        self.target_type = target_type
        self.parameters = parameters
        self.window_title = window_title
        self.camera = camera
        self.displayed_points = []
        # move image so callbacks run in correct thread
        self.image_queue = queue.Queue(maxsize=1)

        self.samples = []
        self.target = None

    def _create_window(self):
        cv2.namedWindow(self.window_title)

    def _close(self):
        self.camera.set_callback(None)
        cv2.destroyWindow(self.window_title)

    def _process_targets(self, image):
        self.target = self.target_type.find(image)

        if self.target is not None:
            cv2.drawContours(image, [np.int0(self.target)], -1, (255, 0, 255), 3)

    def display_point(self, point3d, color, size=3):
        point2d = self.camera.project_points(np.array([point3d]), np.zeros(3), np.zeros(3))
        self.displayed_points.append((point2d[0], color, size))

    def draw_points(self, frame):
        for point, color, size in self.displayed_points:
            position = tuple(np.int0(point))
            cv2.circle(frame, position, radius=size, color=color, thickness=cv2.FILLED)

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
        self._should_run_pose_acquisition = False

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

                if self._should_run_pose_acquisition:
                    self._run_target_pose_acquisition(frame)
                elif hasattr(self, "axes"):
                    cv2.drawFrameAxes(frame, self.camera.camera_matrix, self.camera.no_distortion, self.axes[0], self.axes[1], 0.01)
                    self.camera.publish_pose(self.axes[0], self.axes[1])
                else:
                    self.camera.publish_no_pose()

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

    def _run_target_pose_acquisition(self, frame):
        if self.target is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(np.array([self.target]), self.target_type.marker_size, self.camera.camera_matrix, self.camera.no_distortion)
            rotation_, _ = cv2.Rodrigues(rvecs[0])
            self.camera.publish_pose(rotation_, tvecs[0, 0])
            cv2.drawFrameAxes(frame, self.camera.camera_matrix, self.camera.no_distortion, rvecs[0], tvecs[0], 0.01)
            self.samples.append((rvecs[0], tvecs[0, 0]))

            if len(self.samples) >= self.parameters.pose_samples:
                self.samples = self.samples[1:]
                positions = np.array([t for r, t in self.samples])
                rvecs = np.array([r for r, t in self.samples])
                position = np.mean(positions, axis=0)
                rvec = np.mean(rvecs, axis=0)
                rotation, _ = cv2.Rodrigues(rvec)
                position_std = np.max(np.std(positions, axis=0))
                rotation_std = np.max(np.std(rvecs, axis=0))

                if position_std < 5e-3 and rotation_std < 5e-2:
                    self._acquired_pose = (rotation, position)
                    self.high_variance = False
                else:
                    self.high_variance = True

    def set_axes(self, pose):
        self.axes = pose

    # Get pose of target
    def acquire_pose(self, timeout=None):
        self._acquired_pose = None
        self.samples = []
        self.high_variance = False

        start = time.time()
        self._should_run_pose_acquisition = True

        while not self.should_stop:
            if timeout is not None:
                elapsed = time.time() - start
                early_timeout = len(self.samples) == 0 and elapsed > 0.25*timeout
                if early_timeout:
                    break
                elif elapsed > timeout:
                    if self.high_variance:
                        print("Discarding sample due to high variance in measured target pose")
                    break

            if self._acquired_pose is not None:
                self._should_run_pose_acquisition = False
                return True, self._acquired_pose

        self._should_run_pose_acquisition = False
        return False, None
