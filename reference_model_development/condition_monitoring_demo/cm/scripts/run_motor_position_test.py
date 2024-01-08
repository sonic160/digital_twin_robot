#!/usr/bin/env python3
# Software License Agreement (BSD License)
#
# Control a motor to turn following a unit-pulse signal. And monitor the response.

import rospy
from cm.msg import msg_cm as RosJointState
import Board, time


if __name__ == '__main__':
    rospy.init_node('position_testing_data_publisher', anonymous=True)

    msg = RosJointState()
    msg.name = ['Target value']
    msg.header.frame_id = 'not_relervant'
    msg.position = [0]
    msg.temperature = [0]
    msg.voltage = [0]

    monitor_pos_pub = rospy.Publisher('/position_monitoring', RosJointState, queue_size=1)

    time.sleep(3) # Sleep for .5 second.
    target_value = 0
    Board.setBusServoPulse(6, target_value, 1000) # Turn motor 6 to 0 degree, using 1s.
    msg.header.stamp = rospy.Time.now()
    msg.position[0] = target_value
    monitor_pos_pub.publish(msg)
    print(msg) 

    time.sleep(3) # 延时0.5s
    target_value = 500
    # Run testing.
    Board.setBusServoPulse(6, 500, 1000)
    msg.header.stamp = rospy.Time.now()
    msg.position[0] = target_value
    monitor_pos_pub.publish(msg)
    time.sleep(3)
    print(msg)
