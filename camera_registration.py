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
import math
import numpy as np
import rospy
import sys
from scipy.spatial.transform import Rotation

from camera import Camera
import convex_hull
import vision_tracking


class CameraRegistrationApplication:
    def __init__(self, arm_name, marker_size, expected_interval, camera):
        self.camera = camera
        self.marker_size = marker_size
        self.expected_interval = expected_interval
        self.arm = psm.PSM(arm_name=arm_name, expected_interval=expected_interval)

    def setup(self):
        self.messages.info("Enabling {}...".format(self.arm.name))
        if not self.arm.enable(5):
            self.messages.error("Failed to enable {} within 10 seconds".format(self.arm.name))
            return False

        self.messages.info("Homing {}...".format(self.arm.name))
        if not self.arm.home(10):
            self.messages.error("Failed to home {} within 10 seconds".format(self.arm.name))
            return False

        self.messages.info("Homing complete\n")
        self.arm.jaw.close().wait()

        return True

    def determine_safe_range_of_motion(self):
        self.messages.info(
            "Release the clutch and move the arm around to establish the area the arm can move in"
        )
        self.messages.info("Press enter or 'd' when done")

        def collect_points(hull_points):
            self.done = False

            while self.ok and not self.done:
                pose = self.arm.measured_jp()
                position = np.array([pose[0], pose[1], pose[2]])

                # make list sparser by ensuring >2mm separation
                euclidean = lambda x: np.array(
                    [math.sin(x[0]) * x[2], math.sin(x[1]) * x[2], math.cos(x[2])]
                )
                distance = lambda a, b: np.linalg.norm(euclidean(a) - euclidean(b))
                if len(hull_points) == 0 or distance(position, hull_points[-1]) > 0.005:
                    hull_points.append(position)

                rospy.sleep(self.expected_interval)

            return hull_points

        hull_points = []

        while True:
            hull_points = collect_points(hull_points)
            if not self.ok:
                return False, None

            hull = convex_hull.convex_hull(hull_points)
            if hull is None:
                self.messages.info("Insufficient range of motion, please continue")
            else:
                break

        self.messages.info("Range of motion displayed in plot, close plot window to continue")
        convex_hull.display_hull(hull)
        return self.ok, hull

    # Make sure target is visible and arm is within range of motion
    def ensure_target_visible(self, safe_range):
        self.done = True # run first check immeditately
        first_check = True

        while self.ok:
            rospy.sleep(0.25)

            if not self.done:
                continue

            jp = np.copy(self.arm.measured_jp())
            visible = self.tracker.is_target_visible(timeout=1)
            in_rom = convex_hull.in_hull(safe_range, jp)

            if not visible:
                self.done = False
                if first_check:
                    self.messages.warn(
                        "\nPlease position arm so ArUco target is visible, facing towards camera, and roughly centered within camera's view\n"
                    )
                    first_check = False
                else:
                    self.messages.warn(
                        "Target is not visible, please re-position. Make sure target is not too close"
                    )
                self.messages.info("Press enter or 'd' when done")
            elif not in_rom:
                self.done = False
                self.messages.warn(
                    "Arm is not within user supplied range of motion, please re-position"
                )
            else:
                return True, jp

        return False, None

    # From starting position within view of camera, determine the camera's
    # field of view via exploration while staying within safe range of motion
    # Once field of view is found, collect additional pose samples
    def collect_data(self, safe_range, start_jp, edge_samples=4):
        current_jp = np.copy(start_jp)
        current_jp[4:6] = np.zeros(2)

        target_poses = []
        robot_poses = []

        self.arm.jaw.close()

        def measure_pose(joint_pose):
            nonlocal target_poses
            nonlocal robot_poses

            if not convex_hull.in_hull(safe_range, joint_pose):
                self.messages.error("Safety limit reached!")
                return False

            self.arm.move_jp(joint_pose).wait()
            rospy.sleep(0.5)

            ok, target_pose = self.tracker.acquire_pose(timeout=4.0)
            if not ok:
                return False

            target_poses.append(target_pose)

            pose = self.arm.local.measured_cp().Inverse()
            rotation_quaternion = Rotation.from_quat(pose.M.GetQuaternion())
            rotation = np.float64(rotation_quaternion.as_matrix())
            translation = np.array([pose.p[0], pose.p[1], pose.p[2]], dtype=np.float64)

            robot_poses.append((rotation, np.array(translation)))

            return True

        def bisect_camera_view(pose, ray, min_steps=4, max_steps=6):
            start_pose = np.copy(pose)
            current_pose = np.copy(pose)

            far_limit = convex_hull.intersection(safe_range, start_pose[0:3], ray)
            near_limit = 0.0

            for i in range(max_steps):
                if not self.ok:
                    break

                mid_point = 0.5 * (near_limit + far_limit)
                current_pose[0:3] = start_pose[0:3] + mid_point * ray

                ok = measure_pose(current_pose)
                if ok:
                    near_limit = mid_point
                    self.tracker.display_point(target_poses[-1][1], (255, 0, 255))
                else:
                    far_limit = mid_point

                # Only continue past min_steps if we haven't seen target yet
                if i + 1 >= min_steps and near_limit > 0:
                    break

            end_point = start_pose[0:3] + 0.9 * near_limit * ray
            if len(target_poses) > 0:
                self.tracker.display_point(target_poses[-1][1], (255, 123, 66), size=7)

            return end_point

        def collect(poses, tool_shaft_rotation=math.pi / 10):
            self.messages.progress(0.0)
            for i, pose in enumerate(poses):
                if not self.ok or rospy.is_shutdown():
                    return

                rotation_direction = 1 if i % 2 == 0 else -1
                pose[3] = pose[3] + rotation_direction * tool_shaft_rotation
                shaft_rotations = [
                    pose[3] + rotation_direction * tool_shaft_rotation,
                    pose[3] - rotation_direction * tool_shaft_rotation,
                ]

                for shaft_rotation in shaft_rotations:
                    pose[3] = shaft_rotation
                    ok = measure_pose(pose)
                    if ok:
                        self.tracker.display_point(target_poses[-1][1], (255, 255, 0))
                        break

                self.messages.progress((i+1)/len(sample_poses))

        self.messages.line_break()
        self.messages.info("Determining limits of camera view...")
        self.messages.progress(0.0)
        limits = []

        for axis in range(3):
            ray = np.array([0, 0, 0])
            for direction in [1, -1]:
                if not self.ok:
                    return None

                ray[axis] = direction
                limits.append(bisect_camera_view(current_jp, ray))
                self.messages.progress(len(limits) / 6)
        self.messages.line_break()

        # Limits found above define octahedron, take samples along all 12 edges
        sample_poses = []
        for i in range(len(limits)):
            start = i + 2 if i % 2 == 0 else i + 1
            for j in range(start, len(limits)):
                for t in np.linspace(
                    1 / (edge_samples + 1), 1 - 1 / (edge_samples + 1), edge_samples
                ):
                    pose = np.copy(current_jp)
                    pose[0:3] = limits[j] + t * (limits[i] - limits[j])
                    sample_poses.append(pose)

        self.messages.info("Collecting pose data...")
        collect(sample_poses)
        self.messages.line_break()

        self.messages.info("Data collection complete\n")
        return robot_poses, target_poses

    def compute_registration(self, robot_poses, target_poses):
        error, rotation, translation, gripper = self.camera.calibrate_pose(
            robot_poses, target_poses
        )

        if error < 1e-4:
            self.messages.info("Registration error ({:.3e}) is within normal range".format(error))
        else:
            self.messages.warn(
                "WARNING: registration error ({:.3e}) is unusually high! Should generally be <0.00005".format(
                    error
                )
            )

        distance = np.linalg.norm(translation)
        self.messages.info(
            "Measured distance from RCM to camera origin: {:.3f} m\n".format(distance)
        )

        return self.ok, rotation, translation, gripper

    def save_registration(self, rotation, translation, file_name):
        rotation = rotation.T
        translation = -np.matmul(rotation, translation)

        transform = np.eye(4)
        transform[0:3, 0:3] = rotation
        transform[0:3, 3:4] = translation

        base_frame = {
            "reference-frame": self.tracker.get_camera_frame() or "camera",
            "transform": transform.tolist(),
        }

        output = '"base-frame": {}'.format(json.dumps(base_frame))

        with open(file_name, "w") as f:
            f.write(output)
            f.write("\n")

        self.messages.info("Hand-eye calibration saved to {}".format(file_name))

    # Exit key (q/ESCAPE) handler for GUI
    def _on_quit(self):
        self.ok = False
        self.tracker.stop()
        self.messages.info("\nExiting...")

    # Enter (or 'd') handler for GUI
    def _on_enter(self):
        self.done = True

    def _init_tracking(self):
        target_type = vision_tracking.ArUcoTarget(
            self.marker_size, cv2.aruco.DICT_4X4_50, [0]
        )
        parameters = vision_tracking.VisionTracker.Parameters(4)
        self.messages = vision_tracking.MessageManager()
        self.tracker = vision_tracking.VisionTracker(
            target_type, self.messages, self.camera, parameters
        )

    def run(self):
        try:
            self.ok = True

            self._init_tracking()
            self.ok = self.ok and self.tracker.start(self._on_enter, self._on_quit)
            if not self.ok:
                return

            self.ok = self.ok and self.setup()
            if not self.ok:
                return

            ok, safe_range = self.determine_safe_range_of_motion()
            if not self.ok or not ok:
                return

            ok, start_jp = self.ensure_target_visible(safe_range)
            if not self.ok or not ok:
                return

            data = self.collect_data(safe_range, start_jp)
            if not self.ok:
                return

            if len(data[0]) <= 10:
                self.messages.error("Not enough pose data, cannot compute registration")
                self.messages.error("Please try again, with more range of motion within camera view")
                return

            ok, rvec, tvec, g = self.compute_registration(*data)
            if not ok:
                return

            self.tracker.stop()

            self.save_registration(
                rvec, tvec, "./{}_registration.json".format(self.arm.name)
            )

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
        "-m",
        "--marker_size",
        type=float,
        required=True,
        help="ArUco marker side length - including black border - in same units as camera calibration",
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
    application = CameraRegistrationApplication(
        args.arm, args.marker_size, args.interval, camera
    )
    application.run()


if __name__ == "__main__":
    main()
