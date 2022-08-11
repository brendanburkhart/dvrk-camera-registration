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
import json
import numpy as np
import rospy
import sys
import math

from camera import Camera
import pose_generator
import vision_tracking


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

    def determine_safe_range_of_motion(self):
        print(
            "Release the clutch and move the arm around to establish the area the arm can move in"
        )
        print("Press enter or 'd' when done")

        def collect_points(hull_points):
            self.done = False

            while self.ok and not self.done:
                pose = self.arm.measured_jp()
                position = np.array([pose[0], pose[1], pose[2]])
                hull_points.append(position)

                rospy.sleep(self.expected_interval)

            return hull_points

        hull_points = []

        while True:
            hull_points = collect_points(hull_points)
            if not self.ok:
                return False, None

            hull = pose_generator.convex_hull(hull_points)
            if hull is None:
                print("Insufficient range of motion, please continue")
            else:
                return self.ok, hull

    # Generate `count`` arm poses within safe range of motion
    def registration_poses(self, safe_range, count=10):
        if safe_range is None:
            return []

        js_points = pose_generator.generate(safe_range, count)

        tool_shaft_rotation = self.arm.measured_jp()[3]

        poses = [np.array([*joints, tool_shaft_rotation, 0, 0, 0]) for joints in js_points]

        return poses

    # From starting position within view of camera, determine the camera's
    # field of view via exploration while staying within safe range of motion
    def explore_camera_view(self, safe_range):
        start_jp = np.copy(self.arm.measured_jp())
        current_jp = np.copy(start_jp)

        image_points = []
        object_points = []

        def explore_axis(pose, axis, direction, sub=False):
            nonlocal image_points
            nonlocal object_points
            current_pose = np.copy(pose)
            step = direction * 0.0025 * math.pi

            while self.ok:
                current_pose[axis] += step
                if not pose_generator.in_hull(safe_range, current_pose):
                    return

                self.arm.move_jp(current_pose).wait()
                rospy.sleep(0.2)

                if not self.tracker.target_visible():
                    return

                ok, image_point = self.tracker.acquire_point(timeout=2.0)
                if not ok:
                    print("Hmm")
                    continue

                image_points.append(image_point)

                position = self.arm.measured_cp().p
                position = np.array([position[0], position[1], position[2]])
                object_points.append(position)

                if sub:
                    explore_axis(current_pose, 0, 1)
                    self.arm.move_jp(current_pose).wait()
                    explore_axis(current_pose, 0, -1)
                    self.arm.move_jp(current_pose).wait()
                    explore_axis(current_pose, 1, 1)
                    self.arm.move_jp(current_pose).wait()
                    explore_axis(current_pose, 1, -1)
                    self.arm.move_jp(current_pose).wait()

        explore_axis(current_jp, 2, 1, sub=True)
        self.arm.move_jp(start_jp).wait()
        explore_axis(current_jp, 2, -1, sub=True)

        return object_points, image_points

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
                    break

                self.arm.move_jp(pose).wait()

                ok, image_point = self.tracker.acquire_point(timeout=4.0)
                if not ok:
                    continue

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
        tracking_parameters = vision_tracking.ObjectTracker.Parameters(max_distance=0.08, unique_target=True)
        object_tracker = vision_tracking.ObjectTracker(tracking_parameters)
        target_type = vision_tracking.ArUcoTarget(cv2.aruco.DICT_4X4_50, [0])
        parameters = vision_tracking.VisionTracker.Parameters(2)
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

            self.ok, safe_range = self.determine_safe_range_of_motion()
            if not self.ok:
                return

            # poses = self.registration_poses(safe_range, count=10)
            data = self.explore_camera_view(safe_range)
            if not self.ok:
                return

            ok, rvec, tvec = self.compute_registration(*data)
            if not ok:
                return

            self.tracker.stop()

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
