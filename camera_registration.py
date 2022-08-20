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
import pose_generator
import vision_tracking


class CameraRegistrationApplication:
    def __init__(self, arm_name, marker_size, expected_interval, camera):
        self.camera = camera
        self.marker_size = marker_size
        self.expected_interval = expected_interval
        self.arm = psm.PSM(arm_name=arm_name, expected_interval=expected_interval)

    def setup(self):
        print("Enabling {}...".format(self.arm.name))
        if not self.arm.enable(5):
            print("Failed to enable {} within 10 seconds".format(self.arm.name))
            return False

        print("Homing {}...".format(self.arm.name))
        if not self.arm.home(10):
            print("Failed to home {} within 10 seconds".format(self.arm.name))
            return False

        print("Homing complete\n")
        self.arm.jaw.close().wait()

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

                # make list sparser by ensuring >2mm separation
                euclidean = lambda x: np.array([math.sin(x[0])*x[2], math.sin(x[1])*x[2], math.cos(x[2])])
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

            hull = pose_generator.convex_hull(hull_points)
            if hull is None:
                print("Insufficient range of motion, please continue")
            else:
                return self.ok, hull

    # From starting position within view of camera, determine the camera's
    # field of view via exploration while staying within safe range of motion
    def explore_camera_view(self, safe_range):
        start_jp = np.copy(self.arm.measured_jp())
        current_jp = np.copy(start_jp)
        current_jp[4:6] = np.zeros(2)

        shaft_rotation = current_jp[3]

        target_poses = []
        robot_poses = []

        print("Range of motion displayed in plot, close plot to continue")
        pose_generator.display_hull(safe_range)

        if not pose_generator.in_hull(safe_range, current_jp):
            current_jp[0:3] = pose_generator.centroid(safe_range)
            print("WARNING: starting pose not within safe range of motion, moving to ROM centroid")

        print()

        def collect_point():
            nonlocal target_poses
            nonlocal robot_poses
            
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

            far_limit = pose_generator.intersection(safe_range, start_pose[0:3], ray)
            near_limit = 0.0

            for i in range(max_steps):
                if not self.ok:
                    break

                mid_point = 0.5*(near_limit + far_limit)
                current_pose[0:3] = start_pose[0:3] + mid_point*ray
                if not pose_generator.in_hull(safe_range, current_pose):
                    print("Safety limit reached!")
                    far_limit = mid_point
                    continue

                self.arm.move_jp(current_pose).wait()
                rospy.sleep(0.05)

                ok = collect_point()
                if ok:
                    near_limit = mid_point
                    self.tracker.display_point(target_poses[-1][1], (255, 0, 255))
                else:
                    far_limit = mid_point

                # Only continue past min_steps if we haven't seen target yet
                if i + 1 >= min_steps and near_limit > 0:
                    break

            end_point = start_pose[0:3] + 0.9*near_limit*ray
            if len(target_poses) > 0:
                self.tracker.display_point(target_poses[-1][1], (255, 123, 66), size=7)

            return end_point

        def collect(safe_range, pose):
            if not pose_generator.in_hull(safe_range, pose):
                print("Safety limit reached!")
                return

            ok = collect_point()
            if ok:
                self.tracker.display_point(target_poses[-1][1], (255, 255, 0))
            else:
                self.tracker.display_point(target_poses[-1][1], (0, 0, 255))

        print("Determining limits of camera view...")
        print("Progress: 0%", end="\r")
        limits = []

        for axis in range(3):
            ray = np.array([0, 0, 0])
            for direction in [1, -1]:
                ray[axis] = direction
                limits.append(bisect_camera_view(current_jp, ray))
                print("Progress: {}%".format(int(100*len(limits)/6)), end="\r")
        print("\n")

        print("Collecting pose data...")
        print("Progress: 0%", end="\r")
        sample_poses = []
        tool_shaft_rotation = math.pi/10
        for i in range(len(limits)):
            start = i + 2 if i % 2 == 0 else i + 1 
            for j in range(start, len(limits)):
                a = self.arm.forward_kinematics(limits[i])
                b = self.arm.forward_kinematics(limits[j])

                size = max(0.01, np.linalg.norm(a - b))
                size = min(size, 0.08)
                size = (size-0.01)/0.06
                samples = round(3*size + 3)

                for t in np.linspace(1/(samples+1), 1-1/(samples+1), samples):
                    pose = np.copy(current_jp)
                    pose[0:3] = limits[j] + t*(limits[i] - limits[j])
                    pose[3] = current_jp + tool_shaft_rotation
                    sample_poses.append(np.copy(pose))
                    pose[3] = current_jp - tool_shaft_rotation
                    sample_poses.append(np.copy(pose))

        for i, pose in enumerate(sample_poses):
            collect(safe_range, pose)
            s += samples
            print("Progress: {}%".format(int(100*(i+1)/len(sample_poses))), end="\r")
        print("\n")

        print("Data collection complete\n")
        return robot_poses, target_poses

    def compute_registration(self, robot_poses, target_poses):
        error, rotation, translation = self.camera.calibrate_pose(
            robot_poses, target_poses
        )

        if error < 1e-4:
            print("Registration error ({:.3e}) is within normal range".format(error))
        else:
            print("WARNING: registration error ({:.3e}) is unusually high! Should generally be <0.00005".format(error))

        distance = np.linalg.norm(translation)
        print("Measured distance from RCM to camera origin: {:.3f} m\n".format(distance))

        # def to_homogenous(rotation, translation):
        #     X = np.eye(4)
        #     X[0:3, 0:3] = rotation
        #     X[0:3, 3] = translation.reshape((3,))
        #     return X

        # def to_matrix(pose):
        #     rotation_quaternion = Rotation.from_quat(pose.M.GetQuaternion())
        #     _rotation = np.float64(rotation_quaternion.as_matrix())
        #     translation = np.array([pose.p[0], pose.p[1], pose.p[2]], dtype=np.float64)
        #     return to_homogenous(_rotation, translation)

        # AX = to_homogenous(robot_poses[0][0], robot_poses[0][1])
        # BX = to_homogenous(rotation, translation)
        # CX = to_homogenous(target_poses[0][0], target_poses[0][1])
        # TX = np.matmul(AX, np.matmul(BX, CX))

        # self.done = False
        # print("Press enter or 'd' to continue")
        # while not self.done and self.ok:
        #     rospy.sleep(0.25)
        #     A = to_matrix(self.arm.local.measured_cp())
        #     B = to_homogenous(rotation, translation)
        #     t = np.matmul(np.linalg.inv(B), np.matmul(A, TX))

        #     self.tracker.set_axes((t[0:3, 0:3], t[0:3, 3]))

        return self.ok, rotation, translation

    def save_registration(self, rotation, translation, file_name):
        rotation = np.linalg.inv(rotation)
        translation = -np.matmul(rotation, translation)

        transform = np.eye(4)
        transform[0:3, 0:3] = rotation
        transform[0:3, 3:4] = translation

        base_frame = {
            "reference-frame": self.tracker.get_camera_frame() or "camera",
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
        target_type = vision_tracking.ArUcoTarget(self.marker_size, cv2.aruco.DICT_4X4_50, [0])
        parameters = vision_tracking.VisionTracker.Parameters(4)
        self.tracker = vision_tracking.VisionTracker(
            target_type, self.camera, parameters
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

            print("\nPlease position arm so ArUco target is visible, facing towards camera, and roughly centered within camera's view\n")

            data = self.explore_camera_view(safe_range)
            if not self.ok:
                return

            if len(data[0]) <= 10:
                print("Not enough pose data, cannot compute registration")
                print("Please try again, with more range of motion within camera view")
                return

            ok, rvec, tvec = self.compute_registration(*data)
            if not ok:
                return

            self.tracker.stop()

            self.save_registration(rvec, tvec, "./{}_registration.json".format(self.arm.name))

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
    application = CameraRegistrationApplication(args.arm, args.marker_size, args.interval, camera)
    application.run()


if __name__ == "__main__":
    main()
