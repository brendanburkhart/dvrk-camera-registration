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
from geometry_msgs.msg import Pose, Point, Quaternion
import itertools
import json
import math
import numpy as np
import os
import PyKDL
import rospy
import scipy.spatial
import sys
import xml.etree.ElementTree as ET

import cisst_msgs.srv

import vision_tracking
from camera_calibration import CameraCalibration


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
        self.expected_interval = expected_interval
        self.cartesian_insertion_minimum = 0.055
        self.arm = dvrk.psm(arm_name=robot_name, expected_interval=expected_interval)

        return True

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
        identity = Pose(Point(0.0, 0.0, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0))
        base_frame_topic = "/{}/set_base_frame".format(self.arm.namespace())
        self.set_base_frame_pub = rospy.Publisher(base_frame_topic, Pose, queue_size=1, latch=True)
        self.set_base_frame_pub.publish(identity)
        print("Base frame cleared\n")

        return True

    def determine_safe_range_of_motion(self, valid_points_counter):
        print("Release the clutch and move the arm around to establish the area the arm can move in")
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
                    print("Points within range: {} (want >10, more is better)".format(count), end="\r")

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

    # instrument needs to be inserted past cannula to use Cartesian commands,
    # this will move instrument if necessary so Cartesian commands can be used
    def enter_cartesian_space(self):
        pose = np.copy(self.arm.measured_jp())
        if pose[2] >= self.cartesian_insertion_minimum:
            return True

        pose[2] = self.cartesian_insertion_minimum
        self.arm.move_jp(pose).wait()
        return True

    # return arm to upright position
    def center_arm(self):
        pose = np.copy(self.arm.measured_jp())
        pose.fill(0.0)
        pose[2] = 0.05
        self.arm.move_jp(pose).wait()
        return True

    # Generate series of arm poses within safe rang of motion
    # range_of_motion = (depth, radius, center) describes a
    #     cone with tip at RCM, base centered at (center, depth)
    def registration_poses(self, slices=6, rom=math.pi/4, max_depth=0.17):
        query_cp_name = "{}/local/query_cp".format(self.arm.namespace()) 
        local_query_cp = rospy.ServiceProxy(query_cp_name, cisst_msgs.srv.QueryForwardKinematics)

        # Scale to keep point density equal as depth varies
        scale_rom = lambda depth: math.atan((max_depth/depth)*math.tan(rom))

        def merge_coordinates(alpha, betas, depth):
            alphas = np.repeat(alpha, slices)
            depths = np.repeat(depth, slices)
            return np.column_stack([alphas, betas, depths])

        js_points = []
        depths = np.linspace(max_depth, self.cartesian_insertion_minimum, slices)
        for i, depth in enumerate(depths):
            parity = 1 if i % 2 == 0 else -1
            theta = scale_rom(depth)*parity
            alphas = np.linspace(-theta, theta, slices)
            # Alternate direction so robot follows shortest path
            for i, alpha in enumerate(alphas):
                parity = 1 if i % 2 == 0 else -1
                betas = np.linspace(-parity*theta, parity*theta, slices)
                js_points.extend(merge_coordinates(alpha, betas, depth))

        # We generated square grid, crop to circle so that overall angle
        # stays within specified range of motion
        js_points = [p for p in js_points if (p[0]**2 + p[1]**2) <= rom**2]

        cs_points = []
        for point in js_points:
            # query forward kinematics to get equivalent Cartesian point
            kinematics_request = cisst_msgs.srv.QueryForwardKinematicsRequest()
            kinematics_request.jp.position = [point[0], point[1], point[2], 0.0, 0.0, 0.0, 0.0, 0.0]
            response = local_query_cp(kinematics_request)
            point = response.cp.pose.position
            cs_points.append(np.array([point.x, point.y, point.z]))

        goal_orientation= self.arm.measured_cp().M
        points = [PyKDL.Vector(p[0], p[1], p[2]) for p in cs_points]
        poses = [PyKDL.Frame(goal_orientation, p) for p in points]

        return poses

    # Generates slices^3 poses total
    def _registration_poses(self, range_of_motion, slices=5, size=0.1, depth=0.2):
        offset = np.array([-0.5 * size, -0.5 * size, -depth])
        scale = size / (slices - 1)
        cube_points = np.mgrid[0:slices, 0:slices, 0:slices].T.reshape(-1, 3)
        cube_points = scale * cube_points + offset

        in_hull = range_of_motion.find_simplex(cube_points) >= 0
        points = cube_points[in_hull]

        # Preserve current tool orientation
        goal_rotation = self.arm.measured_cp().M

        points = [PyKDL.Vector(p[0], p[1], p[2]) for p in points]
        poses = [PyKDL.Frame(goal_rotation, p) for p in points]

        return poses

    # move arm to each goal pose, and measure both robot and camera relative positions
    def collect_data(self, poses):
        ok = self.enter_cartesian_space()
        if not ok:
            return False

        target_shift = np.array([0, 0, -0.035])

        try:
            # Slow down arm so vision tracker doesn't lose target
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
        ok, error, rvec, tvec = self.camera_calibration.get_pose(object_points, image_points)
        print("Registration error: {} px".format(error))

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
        _rotation, _ = cv2.Rodrigues(rvec)

        rotation = _rotation.T

        transform = np.eye(4)
        transform[0:3, 0:3] = _rotation
        transform[0:3, 3:4] = tvec
        
        #print(transform)

        #angle = np.linalg.norm(rvec)
        #axis = rvec.reshape((-1,)) / angle

        #alpha = 0.5 * angle;
    
        #q_rotation = Quaternion(axis[0]*math.sin(alpha), axis[1]*math.sin(alpha), axis[2]*math.sin(alpha), math.cos(alpha))
        #pykdl_tf = Pose(Point(tvec[0], tvec[1], tvec[2]), q_rotation)
        #self.set_base_frame_pub.publish(pykdl_tf)
               
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

    def configure_calibration(self, camera_matrix_file, distortion_coefs_file):
        self.camera_matrix = np.load(camera_matrix_file)
        self.distortion_coefs = np.load(distortion_coefs_file)

        self.camera_calibration = CameraCalibration(
            self.camera_matrix, self.distortion_coefs
        )

    def _init_tracking(self):
        object_tracker = vision_tracking.ObjectTracking(15, 200)
        target_type = vision_tracking.ArUcoTarget(cv2.aruco.DICT_4X4_50, [0])
        parameters = vision_tracking.VisionTracker.Parameters(20)
        self.tracker = vision_tracking.VisionTracker(object_tracker, target_type, parameters, self.camera_calibration)

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

            #counter = lambda rom: len(self.registration_poses(rom, 5))
            #self.ok, safe_range = self.determine_safe_range_of_motion(counter)
            if not self.ok:
                return

            poses = self.registration_poses()
            for i in range(1):
                self.setup()
                data = self.collect_data(poses)
                if not self.ok:
                    return

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

