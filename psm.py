#  Author(s):  Anton Deguet, Brendan Burkhart
#  Created on: 2022-08-06

#  (C) Copyright 2022 Johns Hopkins University (JHU), All Rights Reserved.

# --- begin cisst license - do not edit ---

# This software is provided "as is" under an open source license, with
# no warranty.  The complete license can be found in license.txt and
# http://www.cisst.org/cisst/license.txt.

# --- end cisst license ---

from dvrk.arm import *

from cisst_msgs.srv import QueryForwardKinematics, QueryForwardKinematicsRequest
import rospy
import numpy
from geometry_msgs.msg import Pose, Point, Quaternion
import numpy as np

class PSM(arm):
    # initialize the robot
    def __init__(self, arm_name, ros_namespace = "", expected_interval = 0.01):
        self._arm__init_arm(arm_name, ros_namespace, expected_interval)

        query_cp_name = "{}/local/query_cp".format(self.namespace())
        self.local_query_cp = rospy.ServiceProxy(query_cp_name, QueryForwardKinematics)

        base_frame_topic = "/{}/set_base_frame".format(self.namespace())
        self._set_base_frame_pub = rospy.Publisher(
            base_frame_topic, Pose, queue_size=1, latch=True
        )

        # Base class will unregister pub_list on shutdown
        self._arm__pub_list.append(self._set_base_frame_pub)
 
        self.cartesian_insertion_minimum = 0.055

    # Sets speed ratio for move_cp/move_jp
    def set_speed(self, speed):
        self.trajectory_j_set_ratio(speed)

    def clear_base_frame(self):
        identity = Pose(Point(0.0, 0.0, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0))
        self._set_base_frame_pub.publish(identity)

    def forward_kinematics(self, joint_position):
        pad_length = max(0, 8-len(joint_position))
        request = QueryForwardKinematicsRequest()
        request.jp.position = np.pad(joint_position, (0, pad_length)).tolist()
        response = self.local_query_cp(request)
        point = response.cp.pose.position
        return np.array([point.x, point.y, point.z])

    # Bring arm back to center
    def center(self):
        pose = np.copy(self.measured_jp())
        pose.fill(0.0)
        pose[2] = self.cartesian_insertion_minimum
        return self.move_jp(pose)

    # Make sure tool is inserted past cannula so move_cp works
    def enter_cartesian_space(self):
        pose = np.copy(self.measured_jp())
        if pose[2] >= self.cartesian_insertion_minimum:
            class NoWaitHandle:
                def wait(self): pass
                def is_busy(self): return False

            return NoWaitHandle()

        pose[2] = self.cartesian_insertion_minimum
        return self.move_jp(pose)

