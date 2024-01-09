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

    time.sleep(3) # Sleep for 3 second.
    
    # Set target value.
    target_value = 550
    duration = 50
    Board.setBusServoPulse(6, target_value, duration) # Turn motor 6 to 0 degree, using 1s.
    # Publish the target value.
    msg.header.stamp = rospy.Time.now()
    msg.position[0] = target_value
    monitor_pos_pub.publish(msg)
    rospy.loginfo('Publush position command: Target value {}'.format(target_value))

    time.sleep(3) # 延时0.5s
    target_value = 500
    # Run testing.
    Board.setBusServoPulse(6, target_value, duration)
    msg.header.stamp = rospy.Time.now()
    msg.position[0] = target_value
    monitor_pos_pub.publish(msg)
    time.sleep(3)
    rospy.loginfo('Publush position command: Target value {}'.format(target_value))
