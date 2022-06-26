#!/usr/bin/env python

# Author: Brendan Burkhart
# Date: 2022-06-16

# (C) Copyright 2022 Johns Hopkins University (JHU), All Rights Reserved.

# --- begin cisst license - do not edit ---

# This software is provided "as is" under an open source license, with
# no warranty.  The complete license can be found in license.txt and
# http://www.cisst.org/cisst/license.txt.

# --- end cisst license ---

import argparse
import cv2
import dvrk
import json
import math
import numpy as np
import os
import PyKDL
import rospy
import sys
import xml.etree.ElementTree as ET

import blob_tracking
from camera_calibration import CameraCalibration
import direct_linear_transformation as DLT


class CameraRegistrationApplication:
    def configure(self, robot_name, config_file, expected_interval):
        # check that the config file is good
        if not os.path.exists(config_file):
            print('Config file "{:s}" not found'.format(config_file))
            return False

        self.tree = ET.parse(config_file)
        root = self.tree.getroot()

        xpath_search_results = root.findall("./Robot")
        if len(xpath_search_results) != 1:
            print('Can\'t find "Robot" in config file "{:s}"'.format(config_file))
            return False

        xmlRobot = xpath_search_results[0]
        # Verify robot name
        if xmlRobot.get("Name") != robot_name:
            print(
                'Found robot "{:s}" instead of "{:s}", are you using the right config file?'.format(
                    xmlRobot.get("Name"), robot_name
                )
            )
            return False

        serial_number = xmlRobot.get("SN")
        print(
            'Successfully found robot "{:s}", serial number {:s} in XML file'.format(
                robot_name, serial_number
            )
        )
        self.arm = dvrk.psm(arm_name=robot_name, expected_interval=expected_interval)

        return True

    def home(self):
        print("Enabling...")
        if not self.arm.enable(10):
            print("Failed to enable within 10 seconds")
            return False

        print("Homing...")
        if not self.arm.home(10):
            print("Failed to home within 10 seconds")
            return False

        print("Homing complete\n")
        return True

    # instrument needs to be inserted past cannula to use Cartesian commands,
    # this will move instrument if necessary so Cartesian commands can be used
    def enter_cartesian_space(self):
        cartesian_insertion_minimum = 0.05

        pose = np.copy(self.arm.measured_jp())
        if pose[2] >= cartesian_insertion_minimum:
            return True

        pose[2] = cartesian_insertion_minimum
        self.arm.move_jp(pose).wait()
        return True

    # return arm to upright position
    def center_arm(self):
        pose = np.copy(self.arm.measured_jp())
        pose.fill(0.0)
        pose[2] = 0.05
        self.arm.move_jp(pose).wait()
        return True

    # Generate arm poses to registration
    # Yields points on cube grid, where cube has side length 'side',
    # cube is shifted down by 'depth', and cube has count x count x count points
    def registration_poses(self, size, depth, count=4):
        offset = np.array([-0.5 * size, -0.5 * size, -depth])
        scale = size / (count - 1)
        cube_points = np.mgrid[0:count, 0:count, 0:count].T.reshape(-1, 3)
        cube_points = scale * cube_points + offset

        # Jaw opening facing straight down, joint 5 axis pointing forward
        goal_rotation = PyKDL.Rotation()
        goal_rotation[1, 1] = -1.0
        goal_rotation[2, 2] = -1.0

        cube_points = [PyKDL.Vector(*cube_points[i, :]) for i in range(count**3)]
        poses = [PyKDL.Frame(goal_rotation, p) for p in cube_points]

        return poses

    # move arm to each goal pose, and measure both robot and camera relative positions
    def collect_data(self, poses):
        ok = self.enter_cartesian_space()
        if not ok:
            return False

        target_shift = np.array([0, 0, -0.04])

        try:
            # Slow down arm so blob tracker doesn't lose target
            self.arm.trajectory_j_set_ratio(0.2)

            object_points = []
            image_points = []

            print("Collecting data: 0% complete", end="\r")

            # Move arm to variety of positions and record image & world coordinates
            for i, pose in enumerate(poses):
                if not self.ok:
                    self.center_arm()
                    break

                self.arm.move_cp(pose).wait()

                self.ok, image_point = self.tracker.acquire_point()
                image_points.append(image_point)

                position = self.arm.measured_cp().p
                position = np.array([position[0], position[1], position[2]])
                object_points.append(position + target_shift)

                progress = (i + 1) / len(poses)
                print(
                    "Collecting data: {}% complete".format(int(100 * progress)),
                    end="\r",
                )

            print("\n")

        finally:
            # Restore normal arm speed
            self.arm.trajectory_j_set_ratio(1.0)

        self.center_arm()
        return object_points, image_points

    def compute_registration(self, object_points, image_points):
        object_points = np.array(object_points, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)
        ok, rvec, tvec = self.camera_calibration.get_pose(object_points, image_points)
        angle = np.linalg.norm(rvec)
        dist = np.linalg.norm(tvec)
        self.tracker.set_robot_axes(rvec, tvec)
        self.tracker.display_points(object_points, rvec, tvec, (255, 0, 255))
        self.tracker.display_points_2d(image_points, (0, 255, 0))
        print()
        print(
            "Rotation angle: {} radians, translation distance: {} m".format(angle, dist)
        )
        rotation_axis = rvec.reshape((-1,)) / np.linalg.norm(rvec)
        translation_axis = tvec.reshape((-1,)) / np.linalg.norm(tvec)
        print("Axes: {}, {}".format(rotation_axis, translation_axis))
        print("Axis alignment: {}".format(abs(np.dot(rotation_axis, translation_axis))))
        print()

        self.done = False
        print("Press enter or 'd' to continue")
        while not self.done and self.ok:
            rospy.sleep(0.25)

        return self.ok, rvec, tvec

    def save_registration(self, rvec, tvec, file_name):
        rotation_matrix, _ = cv2.Rodrigues(rvec)

        # Correct OpenCV y-axis convention?

        world_to_camera = np.zeros((4, 4))
        world_to_camera[0:3, 0:3] = rotation_matrix
        world_to_camera[0:3, 3:4] = tvec
        world_to_camera[3, 3] = 1.0

        base_frame = {
            "reference-frame": "camera",
            "transform": world_to_camera.tolist(),
        }

        with open(file_name, "w") as f:
            json.dump(base_frame, f)
            f.write("\n")

    # Exit key (q/ESCAPE) handler for GUI
    def _on_quit(self):
        self.ok = False
        self.tracker.stop()
        print("\nExiting...")

    # Enter (or 'd') handler for GUI
    def _on_enter(self):
        self.done = True

    def configure_calibration(self, camera_matrix_file, distortion_coefs_file):
        self.camera_matrix = np.load(camera_matrix_file)
        self.distortion_coefs = np.load(distortion_coefs_file)

        self.camera_calibration = CameraCalibration(
            self.camera_matrix, self.distortion_coefs
        )

    # application entry point
    def run(self):
        try:
            self.ok = True

            tracking_parameters = blob_tracking.BlobTracker.Parameters(
                point_history_length=10
            )
            self.tracker = blob_tracking.BlobTracker(
                self.camera_calibration, tracking_parameters
            )
            self.ok = self.tracker.start(self._on_enter, self._on_quit)
            if not self.ok:
                return

            self.home()

            poses = self.registration_poses(0.1, 0.2, 3)
            data = self.collect_data(poses)
            if not self.ok:
                return

            ok, rvec, tvec = self.compute_registration(*data)
            if not ok:
                return

            self.save_registration(rvec, tvec, "./camera_registration.json")

        finally:
            self.tracker.stop()


def main():
    # ros init node so we can use default ros arguments (e.g. __ns:= for namespace)
    rospy.init_node("dvrk_camera_registration", anonymous=True)
    # strip ros arguments
    argv = rospy.myargv(argv=sys.argv)

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--arm",
        type=str,
        required=True,
        choices=["PSM1", "PSM2", "PSM3"],
        help="arm name corresponding to ROS topics without namespace.  Use __ns:= to specify the namespace",
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=float,
        default=0.01,
        help="expected interval in seconds between messages sent by the device",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="arm IO config file, i.e. something like sawRobotIO1394-xwz-12345.xml",
    )
    parser.add_argument(
        "-m",
        "--matrix",
        type=str,
        default="camera_matrix.npy",
        help="file path of camera matrix",
    )
    parser.add_argument(
        "-d",
        "--distortion",
        type=str,
        default="distortion_coefs.npy",
        help="file path of distortion coefficients",
    )
    args = parser.parse_args(argv[1:])  # skip argv[0], script name

    application = CameraRegistrationApplication()
    ok = application.configure(args.arm, args.config, args.interval)
    if not ok:
        return

    application.configure_calibration(args.matrix, args.distortion)
    application.run()


if __name__ == "__main__":
    main()
