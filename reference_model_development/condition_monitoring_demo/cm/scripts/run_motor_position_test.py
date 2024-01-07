#!/usr/bin/env python3
# Software License Agreement (BSD License)
#
# Control a motor to turn following a unit-pulse signal. And monitor the response.


from cm.msg import msg_cm as RosJointState
import Board, time

time.sleep(0.5) # Sleep for .5 second.
Board.setBusServoPulse(6, 0, 1000) # Turn motor 6 to 0 degree, using 1s.
        
time.sleep(1.5) # 延时0.5s
# Run testing.
Board.setBusServoPulse(6, 500, 1000)
time.sleep(1.5)
