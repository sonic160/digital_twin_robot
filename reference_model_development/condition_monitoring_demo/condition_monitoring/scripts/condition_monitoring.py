#!/usr/bin/env python3
# Software License Agreement (BSD License)
#
## Simple talker demo that published std_msgs/Strings messages
## to the 'condition-monitoring' topic

import rospy
from cm.msg import msg_cm as RosJointState
import Board


class JointCMMessage:
    def __init__(self, name, position, temperature, voltage):
        self.name = name
        self.position = position
        self.temperature = temperature
        self.voltage = voltage


class CMDataPublisher:
    def __init__(self):
        rospy.init_node('condition_monitoring_data_publisher', anonymous=True)
        rate = rospy.get_param('~rate', 10)
        r = rospy.Rate(rate)

        self.joints = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'r_joint']
        self.joint_states = {}

        for joint_name in sorted(self.joints):
            self.joint_states[joint_name] = JointCMMessage(joint_name, 0.0, 0.0, 0.0)

        # Start publisher
        self.joint_states_pub = rospy.Publisher('/condition_monitoring', RosJointState, queue_size=1)

        rospy.loginfo("Starting Joint State Publisher at " + str(rate) + "Hz")

        while not rospy.is_shutdown():
            self.condition_monitoring()
            self.publish_joint_states()
            r.sleep()


    def publish_joint_states(self):
        # Construct message & publish joint states
        msg = RosJointState()
        msg.name = []
        msg.position = []
        msg.temperature = []
        msg.voltage = []

        for joint in self.joint_states.values():
            msg.name.append(joint.name)
            msg.position.append(joint.position)
            msg.temperature.append(joint.temperature)
            msg.voltage.append(joint.voltage)

        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'not_relervant'
        self.joint_states_pub.publish(msg)


    def condition_monitoring(self):
        for motor_idx in range(len(self.joints)):
            joint_name = self.joints[motor_idx]

            pos = Board.getBusServoPulse(motor_idx+1) # Position
            temp = Board.getBusServoTemp(motor_idx+1) # Temperature
            voltage = Board.getBusServoVin(motor_idx+1) # Voltage

            js = JointCMMessage(joint_name, pos, temp, voltage)
            self.joint_states[joint_name] = js


if __name__ == '__main__':
    try:
        s = CMDataPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

