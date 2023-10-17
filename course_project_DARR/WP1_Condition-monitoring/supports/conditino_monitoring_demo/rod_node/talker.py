#!/usr/bin/env python3
# Software License Agreement (BSD License)
#
# Copyright (c) 2008, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Revision $Id$

## Simple talker demo that published std_msgs/Strings messages
## to the 'chatter' topic

import rospy
from std_msgs.msg import String

import time
import Board
import numpy as np

def talker():
    pub = rospy.Publisher('chatter', String, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(1) # 10hz
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
        talker()
    except rospy.ROSInterruptException:
        pass
