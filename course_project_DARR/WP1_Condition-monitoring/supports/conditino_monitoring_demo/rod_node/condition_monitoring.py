#!/usr/bin/env python3
# Software License Agreement (BSD License)
#
## Simple talker demo that published std_msgs/Strings messages
## to the 'condition-monitoring' topic

import rospy
from std_msgs.msg import String

import time
import Board
import numpy as np

def condition_monitoring():
    pub = rospy.Publisher('condition_monitoring', String, queue_size=10)
    rospy.init_node('condition_monitoring_robot', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        # hello_str = "hi, world %s" % rospy.get_time()
        
        pos, temp, voltage = getBusServoStatus()
        hello_str = "position: {}\n temperature: {}\n voltage: {}\n \n".format(pos, temp, voltage) 

        rospy.loginfo(hello_str)
        pub.publish(hello_str)
        rate.sleep()


def getBusServoStatus():
    ''' Get the status of the bus servo: Position, temperature and voltage. '''
    # Initial values.
    pos = np.zeros(6)
    temp = np.zeros(6)
    voltage = np.zeros(6)

    # Do a loop to read the status of the six motors.
    for motor_idx in range(6):
        pos[motor_idx] = Board.getBusServoPulse(motor_idx+1) # Position
        temp[motor_idx] = Board.getBusServoTemp(motor_idx+1) # Temperature
        voltage[motor_idx] = Board.getBusServoVin(motor_idx+1) # Voltage

    return pos, temp, voltage


if __name__ == '__main__':
    try:
        condition_monitoring()
    except rospy.ROSInterruptException:
        pass



