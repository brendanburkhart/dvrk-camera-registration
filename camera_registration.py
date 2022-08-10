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
import psm
from geometry_msgs.msg import Pose, Point, Quaternion
import json
import math
import numpy as np
import PyKDL
import rospy
import scipy.spatial
import sys

import cisst_msgs.srv

import vision_tracking
from camera import Camera

class CameraRegistrationApplication:
    def __init__(self, arm_name, expected_interval, target_z_offset, camera):
        self.camera = camera
        self.expected_interval = expected_interval
        self.target_z_offset = target_z_offset
        self.arm = psm.PSM(arm_name=arm_name, expected_interval=expected_interval)

    def setup(self):
        print("Enabling...")
        if not self.arm.enable(10):
            print("Failed to enable within 10 seconds")
            return False

        print("Homing...")
        if not self.arm.home(10):
            print("Failed to home within 10 seconds")
            return False

        print("Homing complete\n")

        # Set base frame transformation to identity
        self.arm.clear_base_frame()
        print("Base frame cleared\n")

        return True

    def determine_safe_range_of_motion(self, valid_points_counter):
        print(
            "Release the clutch and move the arm around to establish the area the arm can move in"
        )
        print("Press enter or 'd' when done")

        safe_points = [np.array([0, 0, 0])]

        def calculate_range_of_motion(points):
            points = np.array(points)

            try:
                hull = scipy.spatial.ConvexHull(points)
            except scipy.spatial.QhullError:
                return None

            hull_points = points[hull.vertices]
            range_of_motion = scipy.spatial.Delaunay(hull_points)
            return range_of_motion

        point_count = -1

        def collect_points():
            nonlocal point_count
            nonlocal safe_points

            self.done = False

            while self.ok and not self.done:
                pose = self.arm.measured_cp()
                position = np.array([pose.p[0], pose.p[1], pose.p[2]])
                safe_points.append(position)

                rom = calculate_range_of_motion(safe_points)
                count = valid_points_counter(rom) if rom is not None else 0
                if count > point_count:
                    point_count = count
                    print(
                        "Points within range: {} (want >10, more is better)".format(
                            count
                        ),
                        end="\r",
                    )

                rospy.sleep(self.expected_interval)

            return self.ok

        while True:
            collect_points()
            if not self.ok:
                return False, None

            rom = calculate_range_of_motion(safe_points)
            count = valid_points_counter(rom) if rom is not None else 0
            if count < 10:
                print("Insufficient range of motion, please continue")
            else:
                break

        return self.ok, rom

    # Generate series of arm poses within safe rang of motion
    # range_of_motion = (depth, radius, center) describes a
    #     cone with tip at RCM, base centered at (center, depth)
    def registration_poses(self, slices=4, rom=math.pi / 4, max_depth=0.17):
        # Scale to keep point density equal as depth varies
        scale_rom = lambda depth: math.atan((max_depth / depth) * math.tan(rom))

        def merge_coordinates(alpha, betas, depth):
            alphas = np.repeat(alpha, slices)
            depths = np.repeat(depth, slices)
            return np.column_stack([alphas, betas, depths])

        js_points = []
        depths = np.linspace(max_depth, self.arm.cartesian_insertion_minimum, slices)
        for i, depth in enumerate(depths):
            parity = 1 if i % 2 == 0 else -1
            theta = scale_rom(depth) * parity
            alphas = np.linspace(-theta, theta, slices)
            # Alternate direction so robot follows shortest path
            for i, alpha in enumerate(alphas):
                parity = 1 if i % 2 == 0 else -1
                betas = np.linspace(-parity * theta, parity * theta, slices)
                js_points.extend(merge_coordinates(alpha, betas, depth))

        # We generated square grid, crop to circle so that overall angle
        # stays within specified range of motion
        js_points = [p for p in js_points if (p[0] ** 2 + p[1] ** 2) <= rom**2]

        # Get cartesian-space equivalents to joint-space poses
        cs_points = [self.arm.forward_kinematics(point) for point in js_points]

        goal_orientation = self.arm.measured_cp().M
        points = [PyKDL.Vector(p[0], p[1], p[2]) for p in cs_points]
        poses = [PyKDL.Frame(goal_orientation, p) for p in points]

        return poses

    # move arm to each goal pose, and measure both robot and camera relative positions
    def collect_data(self, poses):
        self.arm.enter_cartesian_space().wait()
        target_offset = np.array([0, 0, self.target_z_offset])

        try:
            # Slow down arm so vision tracker doesn't lose target
            self.arm.set_speed(0.2)

            object_points = []
            image_points = []

            print("Collecting data: 0% complete", end="\r")

            # Move arm to variety of positions and record image & world coordinates
            for i, pose in enumerate(poses):
                if not self.ok:
                    self.arm.center()
                    break

                self.arm.move_cp(pose).wait()

                self.ok, image_point = self.tracker.acquire_point()
                image_points.append(image_point)

                position = self.arm.measured_cp().p
                position = np.array([position[0], position[1], position[2]])
                object_points.append(position)

                progress = (i + 1) / len(poses)
                print(
                    "Collecting data: {}% complete".format(int(100 * progress)),
                    end="\r",
                )

            print("\n")

        finally:
            # Restore normal arm speed
            self.arm.set_speed(1.0)

        return object_points, image_points

    def compute_registration(self, object_points, image_points):
        object_points = np.array(object_points, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)
        ok, error, rvec, tvec = self.camera.get_pose(
            object_points, image_points
        )
        print("Registration error: {} px".format(error))

        self.tracker.set_robot_axes(rvec, tvec)
        self.tracker.display_points(object_points, rvec, tvec, (255, 0, 255))
        self.tracker.display_points_2d(image_points, (0, 255, 0))

        distance = np.linalg.norm(tvec)
        print("\nTranslation distance: {} m\n".format(distance))

        self.done = False
        print("Press enter or 'd' to continue")
        while not self.done and self.ok:
            rospy.sleep(0.25)

        return self.ok, rvec, tvec

    def save_registration(self, rvec, tvec, file_name):
        rotation, _ = cv2.Rodrigues(rvec)

        transform = np.eye(4)
        transform[0:3, 0:3] = rotation
        transform[0:3, 3:4] = tvec

        base_frame = {
            "reference-frame": "camera",
            "transform": transform.tolist(),
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

    def _init_tracking(self):
        tracking_parameters = vision_tracking.ObjectTracker.Parameters(max_distance=0.08)
        object_tracker = vision_tracking.ObjectTracker(tracking_parameters)
        target_type = vision_tracking.ArUcoTarget(cv2.aruco.DICT_4X4_50, [0])
        parameters = vision_tracking.VisionTracker.Parameters(20)
        self.tracker = vision_tracking.VisionTracker(
            object_tracker, target_type, self.camera, parameters
        )

    def run(self):
        try:
            self.ok = True

            self._init_tracking()
            self.ok = self.tracker.start(self._on_enter, self._on_quit)
            if not self.ok:
                return

            self.ok = self.setup()
            if not self.ok:
                return

            # counter = lambda rom: len(self.registration_poses(rom, 5))
            # self.ok, safe_range = self.determine_safe_range_of_motion(counter)
            if not self.ok:
                return

            poses = self.registration_poses()
            data = self.collect_data(poses)
            if not self.ok:
                return

            self.tracker.stop()

            ok, rvec, tvec = self.compute_registration(*data)
            if not ok:
                return

            self.save_registration(rvec, tvec, "./camera_registration.json")

        finally:
            self.tracker.stop()
            self.arm.unregister()


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
        "-o",
        "--offset",
        type=float,
        default=-0.04,
        help="z-offset of vision target center",
    )
    parser.add_argument(
        "-c",
        "--camera_image_topic",
        type=str,
        required=True,
        help="ROS topic of rectified color image transport",
    )
    parser.add_argument(
        "-t",
        "--camera_info_topic",
        type=str,
        required=True,
        help="ROS topic of camera info for camera",
    )
    args = parser.parse_args(argv[1:])  # skip argv[0], script name

    camera = Camera(args.camera_info_topic, args.camera_image_topic)
    application = CameraRegistrationApplication(args.arm, args.interval, args.offset, camera)
    application.run()


if __name__ == "__main__":
    main()
