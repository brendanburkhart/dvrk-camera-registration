#!/usr/bin/python

# Author: Brendan Burkhart 
# Date: 2022-06-21

# (C) Copyright 2022 Johns Hopkins University (JHU), All Rights Reserved.

# --- begin cisst license - do not edit ---

# This software is provided "as is" under an open source license, with
# no warranty.  The complete license can be found in license.txt and
# http://www.cisst.org/cisst/license.txt.

# --- end cisst license ---

import argparse
import cv2
import numpy as np
import time


class CameraCalibration:
    def __init__(self, camera_matrix, distortion_coef):
        self.window_title = "Camera Correction"
        self.camera_matrix = camera_matrix
        self.distortion_coef = distortion_coef

    def configure_image_size(self, shape):
        h, w = shape[:2]
        alpha = 1.0
        self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.distortion_coef, (w,h), alpha, (w,h))
        self.no_distortion = np.array([], dtype=np.float32)
    
    def undistort(self, image):
        image = cv2.undistort(image, self.camera_matrix, self.distortion_coef, None, self.new_camera_matrix)
        # crop image to ROI 
        x, y, w, h = self.roi
        image = image[y:y+h, x:x+w]

        return image

    def project_points(self, object_points, rodrigues_rotation, translation_vector):
        image_points, _ = cv2.projectPoints(object_points, rodrigues_rotation, translation_vector, self.new_camera_matrix, self.no_distortion)
        
        # account for cropping
        x, y, w, h = self.roi
        # image_points += np.array([-x, -y])
        
        # opencv double-nests the points for some reason, i.e. each point is array([[x, y]])
        image_points = image_points.reshape((-1, 2))

        return image_points

    def get_pose(self, object_points, image_points):
        ok, rotation, translation = cv2.solvePnP(object_points, image_points, self.new_camera_matrix, self.no_distortion)
        if not ok:
            return ok, 0.0, rotation, translation

        projected_points = self.project_points(object_points, rotation, translation)
        reprojection_error = np.mean(np.linalg.norm(image_points - projected_points, axis=1))

        return ok, reprojection_error, rotation, translation 
        

class CalibrationApplication:
    def __init__(self, square_size, grid_shape, min_captures, capture_delay, camera_matrix_file, distortion_coef_file):
        self.window_title = "Camera Calibration"
        self.square_size = square_size
        self.grid_shape = grid_shape
        self.min_captures = min_captures
        self.capture_delay = capture_delay
        self.camera_matrix_file = camera_matrix_file 
        self.distortion_coef_file = distortion_coef_file
        self.grid_points = self._create_grid_points(square_size, grid_shape)

    def _create_grid_points(self, square_size, grid_shape):
        # Create grid with unit squares, where all points lie on Z=0 plane
        points = np.zeros((grid_shape[0]*grid_shape[1], 3), np.float32)
        points[:, :2] = np.mgrid[0:grid_shape[0], 0:grid_shape[1]].T.reshape(-1, 2)
        # Scale grid to specified size
        points = square_size*points
        return points

    def _create_window(self):
         cv2.namedWindow(self.window_title)
 
    def _init_video(self):
        self.video_capture = cv2.VideoCapture(0)
        self.video_capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        ok = False
        if self.video_capture.isOpened():
            ok, _ = self.video_capture.read()
 
        if not ok:
            print("\n\nFailed to initialize camera.")
 
        return ok

    def _compute_calibration(self, object_points, image_points, shape):
        ok, camera_matrix, distortion, rotation, translation = cv2.calibrateCamera(object_points, image_points, shape[::1], None, None)

        total_error = 0.0
        for i in range(len(object_points)):
            projected_points, _ = cv2.projectPoints(object_points[i], rotation[i], translation[i], camera_matrix, distortion)
            total_error += cv2.norm(image_points[i], projected_points, cv2.NORM_L2)/len(object_points[i])

        mean_error = total_error/len(object_points)

        return ok, camera_matrix, distortion, mean_error

    def _capture(self):
        object_points = []
        image_points = []
        captures = 0
        last_capture_time = None

        while captures < self.min_captures:
            ok, frame = self.video_capture.read()
            if not ok:
                print("\n\nFailed to read from camera")
                return False, None, None, None

            # ensure sufficient time between captures for pattern to be moved
            elapsed_time = 0 if last_capture_time is None else time.time() - last_capture_time
            next_capture_ready = last_capture_time is None or elapsed_time > self.capture_delay

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.grid_shape, None)
            if ret and next_capture_ready:
                # refine corner locations to sub-pixel accuracy
                search_window = (6, 6)
                no_zero_zone = (-1, -1)
                termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (6, 6), zeroZone=no_zero_zone, criteria=termination_criteria)
                object_points.append(self.grid_points)
                image_points.append(corners)

                captures += 1
                print("{} captures, need {} more".format(captures, self.min_captures - captures))
                last_capture_time = time.time()

            cv2.drawChessboardCorners(frame, self.grid_shape, corners, ret)
            cv2.imshow(self.window_title, frame)

            key = cv2.waitKey(20)
            key = key & 0xFF # Upper bits are modifiers (control, alt, etc.)
            escape = 27
            if key == ord("q") or key == escape:
                print("\n\nQuitting...")
                return False, None, None, None

        return True, object_points, image_points, gray.shape

    def _display_calibration(self, camera_calibration):
        print("Calibration applied to camera")
        print("Press ENTER/'d' to accept calibration, or ESCAPE/'q' to discard") 

        ok, frame = self.video_capture.read()
        if not ok:
            return False

        camera_calibration.configure_image_size(frame.shape) 

        while True:
            ok, frame = self.video_capture.read()
            if not ok:
                print("\n\nFailed to read from camera")
                return False

            image = camera_calibration.undistort(frame)
            cv2.imshow(self.window_title, image)

            key = cv2.waitKey(20)
            key = key & 0xFF # Upper bits are modifiers (control, alt, etc.)
            escape = 27
            if key == ord("q") or key == escape:
                return False
        
            if key == ord('d') or key == ord('\r') or key == ord('\n'):
                return True

    def run(self):
        self._create_window()
        ok = self._init_video()
        if not ok:
            return

        ok, object_points, image_points, shape = self._capture()
        if not ok:
            return

        ok, camera_matrix, distortion_coefs, error = self._compute_calibration(object_points, image_points, shape)
        if not ok:
            print("Failed to obtain good calibration")
            return

        print("Calibration succeeded, error is: {}".format(error))
        calibration = CameraCalibration(camera_matrix, distortion_coefs)
        
        accept = self._display_calibration(calibration)
        if not accept:
            print("Discarding calibration")
            return

        np.save(self.camera_matrix_file, camera_matrix)
        np.save(self.distortion_coef_file, distortion_coefs)
        print("Calibation saved to {} and {}".format(self.camera_matrix_file, self.distortion_coef_file))


if __name__ == "__main__":
    chessboard_pattern_url = "https://github.com/opencv/opencv/blob/4.x/doc/pattern.png"
    print("Print out a chessboard calibration pattern, e.g. \"{:s}\"".format(chessboard_pattern_url))
    print("By convention, if your pattern has N rows and M columns, it is a (N-1)x(M-1) calibration pattern")
    print("Measure the size of the squares. If you are doing pose estimation, use the same units here")
    print()
    print("While program is running, slowly move pattern around on screen. If pattern is found, it will be")
    print("marked in a rainbow pattern and calibration captures will be taken. Try to cover variety of")
    print("pattern positions and angles, and make sure to get captures near edges of camera view.")

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--size', type=float, required=True,
                        help = "size (in desired units) of each square on the printed chessboard")
    parser.add_argument('-g', '--grid', nargs=2, type=int, required=True,
                        help = "chessboard grid shape (at least 3x3), e.g. '--grid 4 3' for 4x3")
    parser.add_argument('-c', '--captures', type=int, default=15,
                        help = "number of captures to use for calibration, must be >=10")
    parser.add_argument('-i', '--interval', type=int, default=1,
                        help = "how long to wait between captures")
    parser.add_argument('-m', '--matrix', type=str, default="camera_matrix.npy",
                        help = "file path to save camera matrix to")
    parser.add_argument('-d', '--distortion', type=str, default="distortion_coefs.npy",
                        help = "file path to save distortion coefficients to")
    args = parser.parse_args()

    # Validate args
    if args.grid[0] <= 2 or args.grid[1] <= 2:
        parser.error("Minimum chessboard size is 3x3")

    if args.captures < 10:
        parser.error("Minimum captures needed is at least 10")

    app = CalibrationApplication(args.size, tuple(args.grid), args.captures, args.interval, args.matrix, args.distortion)
    app.run()

