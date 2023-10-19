#!/usr/bin/env python
import rospy
from condition_monitoring.msg import msg_cm as RosJointState

def callback(data):
    print(data)


def listener():
    rospy.init_node('cm_listener', anonymous=True)
    rospy.Subscriber('condition_monitoring', RosJointState, callback)
 
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()
 
if __name__ == '__main__':
    listener()
