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
        print(self.camera_matrix)
        print(self.distortion_coef)

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
        image_points += np.array([-x, -y])
        
        # opencv double-nests the points for some reason, i.e. each point is array([[x, y]])
        image_points = image_points.reshape((-1, 2))

        return image_points

    def get_pose(self, object_points, image_points):
        return cv2.solvePnP(object_points, image_points, self.camera_matrix, self.distortion_coef)
        

class CalibrationApplication:
    def __init__(self, camera_matrix_file, distortion_coef_file):
        self.window_title = "Camera Calibration"
        self.camera_matrix_file = camera_matrix_file 
        self.distortion_coef_file = distortion_coef_file

    def _create_window(self):
         cv2.namedWindow(self.window_title)
 
    def _init_video(self):
        self.video_capture = cv2.VideoCapture(0)
        ok = False
        if self.video_capture.isOpened():
            ok, _ = self.video_capture.read()
 
        if not ok:
            print("\n\nFailed to initialize camera.")
 
        return ok

    def _display_calibration(self, camera_calibration):
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
            points = np.array([[-0.1524*0.5, 0, 0.3048], [0.1524*0.5, 0, 0.3048]])
            image_points = camera_calibration.project_points(points, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
            image_points = np.int0(image_points)
            cv2.line(image, tuple(image_points[0]), tuple(image_points[1]), (255, 255, 0), 3)
        
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

        camera_matrix = np.load(self.camera_matrix_file)
        distortion_coefs = np.load(self.distortion_coef_file)
        calibration = CameraCalibration(camera_matrix, distortion_coefs)
        
        self._display_calibration(calibration)


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--matrix', type=str, default="camera_matrix.npy",
                        help = "file path to camera matrix")
    parser.add_argument('-d', '--distortion', type=str, default="distortion_coefs.npy",
                        help = "file path to distortion coefficients")
    args = parser.parse_args()
    
    app = CalibrationApplication(args.matrix, args.distortion)
    app.run()

