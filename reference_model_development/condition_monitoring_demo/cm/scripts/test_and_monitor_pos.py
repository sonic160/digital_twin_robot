#!/usr/bin/env python3
# Software License Agreement (BSD License)
#
# Control a motor to turn following a unit-pulse signal. And monitor the response.

import rospy
from cm.msg import msg_cm as RosJointState
import Board


class CMDataPublisher:
    def __init__(self, freq=10, monitored_motor=6):
        rospy.init_node('condition_monitoring_data_publisher', anonymous=True)
        rate = rospy.get_param('~rate', freq)
        r = rospy.Rate(rate)

        self.msg = RosJointState()
        self.msg.name = ['motor_{}'.format(monitored_motor)]
        self.msg.header.frame_id = 'not_relervant'
        self.msg.position = [0]
        self.msg.temperature = [0]
        self.msg.voltage = [0]

        self.monitored_motor = monitored_motor

        # Start publisher
        self.monitor_pos_pub = rospy.Publisher('/condition_monitoring', RosJointState, queue_size=50)

        rospy.loginfo("Monitoring the position of motor " + str(monitored_motor) + " at " + str(rate) + "Hz")

        while not rospy.is_shutdown():
            self.condition_monitoring()
            r.sleep()

    def condition_monitoring(self):
        # Log current time.
        self.msg.header.stamp = rospy.Time.now()
        # Get CM data.
        motor_idx = self.monitored_motor
        self.msg.position[0] = Board.getBusServoPulse(motor_idx) # Position
        # Publish the data.
        self.monitor_pos_pub.publish(self.msg)


if __name__ == '__main__':
    try:
        s = CMDataPublisher(freq=100, monitored_motor=6)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

