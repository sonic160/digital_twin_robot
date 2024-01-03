#!/usr/bin/env python3
# Software License Agreement (BSD License)
#
## Simple talker demo that published std_msgs/Strings messages
## to the 'condition-monitoring' topic

import rospy
from cm.msg import msg_cm as RosJointState
import Board


class CMDataPublisher:
    def __init__(self, freq=10):
        rospy.init_node('condition_monitoring_data_publisher', anonymous=True)
        rate = rospy.get_param('~rate', freq)
        r = rospy.Rate(rate)

        self.msg = RosJointState()
        self.msg.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'r_joint']
        self.msg.header.frame_id = 'not_relervant'
        self.msg.position = [0]*6
        self.msg.temperature = [0]*6
        self.msg.voltage = [0]*6

        # Start publisher
        self.joint_states_pub = rospy.Publisher('/condition_monitoring', RosJointState, queue_size=50)

        rospy.loginfo("Starting Joint State Publisher at " + str(rate) + "Hz")

        while not rospy.is_shutdown():
            self.condition_monitoring()
            r.sleep()

    def condition_monitoring(self):
        # Log current time.
        self.msg.header.stamp = rospy.Time.now()
        # Get CM data.
        for motor_idx in range(6):
            self.msg.position[motor_idx] = Board.getBusServoPulse(motor_idx+1) # Position
            self.msg.temperature[motor_idx] = Board.getBusServoTemp(motor_idx+1) # Temperature
            self.msg.voltage[motor_idx] = Board.getBusServoVin(motor_idx+1) # Voltage
        # Publish the data.
        self.joint_states_pub.publish(self.msg)


if __name__ == '__main__':
    try:
        s = CMDataPublisher(10)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

