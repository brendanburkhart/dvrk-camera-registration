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
import dvrk
import math
import numpy as np
import os
import PyKDL
import rospy
import sys
import xml.etree.ElementTree as ET

import blob_tracking
from camera_calibration import CameraCalibration

class CameraRegistrationApplication:
    def configure(self, robot_name, config_file, expected_interval):
        # check that the config file is good
        if not os.path.exists(config_file):
            print("Config file \"{:s}\" not found".format(config_file))
            return False

        self.tree = ET.parse(config_file)
        root = self.tree.getroot()

        xpath_search_results = root.findall("./Robot")
        if len(xpath_search_results) != 1:
            print("Can't find \"Robot\" in config file \"{:s}\"".format(config_file))
            return False
        
        xmlRobot = xpath_search_results[0]
        # Verify robot name
        if xmlRobot.get("Name") != robot_name:
            print("Found robot \"{:s}\" instead of \"{:s}\", are you using the right config file?".format(xmlRobot.get("Name"), robot_name))
            return False

        serial_number = xmlRobot.get("SN")
        print("Successfully found robot \"{:s}\", serial number {:s} in XML file".format(robot_name, serial_number))
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
    def registration_poses(self, size, depth, count=3):
        cube_points = size/(count-1)*np.mgrid[0:count, 0:count, 0:count].T.reshape(-1, 3) + np.array([-0.5*size, -0.5*size, -depth])
    
        # Jaw opening facing straight down, joint 5 axis pointing forward
        goal_orientation = PyKDL.Rotation()
        goal_orientation[1,1] = -1.0
        goal_orientation[2,2] = -1.0
 
        pose = self.arm.measured_cp()
        pose.M = goal_orientation

        for i in range(count**3):
            position = PyKDL.Vector(*cube_points[i, :])
            pose.p = position
            yield pose

    def collect_data(self, size, depth):
        ok = self.enter_cartesian_space()
        if not ok:
            return False

        try:
            # Slow down arm so blob tracker doesn't lose target
            self.arm.trajectory_j_set_ratio(0.2)

            object_points = []
            image_points = []
   
            # Move arm to variety of positions and record image & world coordinates 
            poses = self.registration_poses(size, depth)
            for pose in poses:
                if not self.ok:
                    self.center_arm()
                    break

                self.arm.move_cp(pose).wait()
                
                position = self.arm.measured_cp().p
                object_points.append(np.array([position[0], position[1], position[2]]))
                
                self.ok, image_point = self.tracker.acquire_point()
                image_points.append(image_point)

        finally:
            # Restore normal arm speed
            self.arm.trajectory_j_set_ratio(1.0)

        self.center_arm()
        return object_points, image_points
            
    def compute_registration(self, object_points, image_points):
        object_points = np.array(object_points, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)
        ok, rvec, tvec = self.camera_calibration.get_pose(object_points, image_points)
        print(ok)
        angle = np.linalg.norm(rvec)
        dist = np.linalg.norm(tvec)
        self.tracker.set_robot_axes(rvec, tvec)
        self.tracker.display_points(object_points, rvec, tvec, (255, 0, 255))
        self.tracker.display_points_2d(image_points, (0, 255, 0))
        print()
        print("Rotation angle: {} radians, translation distance: {} m".format(angle, dist))
        rotation_axis = rvec.reshape((-1,))/np.linalg.norm(rvec)
        translation_axis = tvec.reshape((-1,))/np.linalg.norm(tvec)
        print("Axes: {}, {}".format(rotation_axis, translation_axis))
        print("Axis alignment: {}".format(abs(np.dot(rotation_axis, translation_axis))))
        print()
        
        self.done = False
        print("Press enter or 'd' to continue")
        while not self.done and self.ok:
            rospy.sleep(0.25)
       
        return rvec, tvec

    # Exit key (q/ESCAPE) handler for GUI
    def _on_quit(self):
        self.ok = False
        self.tracker.stop()
        print("Exiting...")

    # Enter (or 'd') handler for GUI
    def _on_enter(self):
        self.done = True

    def configure_calibration(self, camera_matrix_file, distortion_coefs_file):
        self.camera_matrix = np.load(camera_matrix_file)
        self.distortion_coefs = np.load(distortion_coefs_file)

        self.camera_calibration = CameraCalibration(self.camera_matrix, self.distortion_coefs)

    # application entry point
    def run(self):
        try:
            self.ok = True
            
            self.tracker = blob_tracking.BlobTracker(self.camera_calibration)
            self.tracker.start(self._on_enter, self._on_quit)

            self.home()
            data = self.collect_data(0.1, 0.2)
            if self.ok:
                registration = self.compute_registration(*data)

        finally:
            self.tracker.stop()


def main():
    # ros init node so we can use default ros arguments (e.g. __ns:= for namespace)
    rospy.init_node('dvrk_camera_registration', anonymous=True)
    # strip ros arguments
    argv = rospy.myargv(argv=sys.argv)

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--arm', type=str, required=True,
                        choices=['PSM1', 'PSM2', 'PSM3'],
                        help = 'arm name corresponding to ROS topics without namespace.  Use __ns:= to specify the namespace')
    parser.add_argument('-i', '--interval', type=float, default=0.01,
                        help = 'expected interval in seconds between messages sent by the device')
    parser.add_argument('-c', '--config', type=str, required=True,
                        help = 'arm IO config file, i.e. something like sawRobotIO1394-xwz-12345.xml')
    parser.add_argument('-m', '--matrix', type=str, default="camera_matrix.npy",
                        help = "file path of camera matrix")
    parser.add_argument('-d', '--distortion', type=str, default="distortion_coefs.npy",
                        help = "file path of distortion coefficients")
    args = parser.parse_args(argv[1:]) # skip argv[0], script name

    application = CameraRegistrationApplication()
    ok = application.configure(args.arm, args.config, args.interval)
    if not ok:
        return

    application.configure_calibration(args.matrix, args.distortion)
    application.run()

if __name__ == '__main__':
    main()
   