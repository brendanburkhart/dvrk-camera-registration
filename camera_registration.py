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

        target_poses = []
        robot_poses = []

        pose_generator.display_hull(safe_range)

        def collect_point():
            nonlocal target_poses
            nonlocal robot_poses
            
            ok, target_pose = self.tracker.acquire_pose(timeout=4.0)
            if not ok:
                return False

            target_poses.append(target_pose)

            pose = self.arm.measured_cp()
            rotation_quaternion = Rotation.from_quat(pose.M.GetQuaternion())
            rotation = rotation_quaternion.as_matrix()
            translation = np.array([pose.p[0], pose.p[1], pose.p[2]])

            robot_poses.append((rotation, np.array(translation)))

            return True

        def bisect_camera_view(pose, ray, max_steps=5):
            start_pose = np.copy(pose)
            current_pose = np.copy(pose)

            far_limit = pose_generator.intersection(safe_range, start_pose[0:3], ray)
            near_limit = 0.0

            for _ in range(max_steps):
                if not self.ok:
                    break

                mid_point = 0.5*(near_limit + far_limit)
                current_pose[0:3] = start_pose[0:3] + mid_point*ray
                if not pose_generator.in_hull(safe_range, current_pose):
                    print("Safety limit reached!")
                    return

                self.arm.move_jp(current_pose).wait()
                rospy.sleep(0.5)

                ok = collect_point()
                if ok:
                    near_limit = mid_point
                else:
                    far_limit = mid_point


            return start_pose[0:3] + 0.8*near_limit*ray

        def collect(safe_range, start, end, pose, steps=3):
            current_pose = np.copy(pose)
            
            for t in np.linspace(1/(steps+1), 1-1/(steps+1), steps):
                if not self.ok:
                    break
                
                current_pose[0:3] = start + t*(end - start)
                if not pose_generator.in_hull(safe_range, current_pose):
                    print("Safety limit reached!")
                    return

                self.arm.move_jp(current_pose).wait()
                rospy.sleep(0.5)

                collect_point()

        def explore_axis(pose, axis):
            nonlocal target_poses
            nonlocal robot_poses
            ray = np.array([0, 0, 0])
            
            ray[axis] = 1
            limit1 = bisect_camera_view(pose, ray)
            ray[axis] = -1
            limit2 = bisect_camera_view(pose, ray)
            return [limit1, limit2]

        limits = []
        limits.extend(explore_axis(current_jp, 0))
        limits.extend(explore_axis(current_jp, 1))
        limits.extend(explore_axis(current_jp, 2))

        print(len(robot_poses))

        for i in range(len(limits)):
            start = i + 2 if i % 2 == 0 else i + 1 
            for j in range(start, len(limits)):
                collect(safe_range, limits[i], limits[j], current_jp)

        print(len(robot_poses))

        return robot_poses, target_poses

    def compute_registration(self, robot_poses, target_poses):
        error, rotation, translation = self.camera.calibrate_pose(
            robot_poses, target_poses
        )
        print("\nRegistration error: {}".format(error))

        distance = np.linalg.norm(translation)
        print("\nTranslation distance: {} m\n".format(distance))

        self.done = False
        print("Press enter or 'd' to continue")
        while not self.done and self.ok:
            rospy.sleep(0.25)

        return self.ok, rotation, translation

    def save_registration(self, rotation, translation, file_name):
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
        target_type = vision_tracking.ArUcoTarget(0.01, cv2.aruco.DICT_4X4_50, [0])
        parameters = vision_tracking.VisionTracker.Parameters(5)
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

            # poses = self.registration_poses(safe_range, count=10)
            data = self.explore_camera_view(safe_range)
            if not self.ok:
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
