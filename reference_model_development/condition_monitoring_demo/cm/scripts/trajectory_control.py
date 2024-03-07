#!/usr/bin/env python3
# Software License Agreement (BSD License)
# 
# This script controls a motor to turn following a unit-pulse signal. And monitor the response.


import threading, rospy, Board, time
from cm.msg import msg_cm as RosJointState
import argparse


class CMDataPublisher:
    def __init__(self, node, freq=10):
        self.node = node
        rate = self.node.get_param('~rate', freq)
        self.r = rospy.Rate(rate)

        self.msg = RosJointState()
        self.msg.name = ['motor position']
        self.msg.header.frame_id = 'not_relervant'
        self.msg.position = [0, 0, 0, 0, 0, 0]
        self.msg.temperature = [0, 0, 0, 0, 0, 0]
        self.msg.voltage = [0, 0, 0, 0, 0, 0]

        # Start publisher
        self.monitor_pos_pub = rospy.Publisher('/condition_monitoring', RosJointState, queue_size=50)
        rospy.loginfo("Monitoring the position of motors" + " at " + str(rate) + "Hz")

    def get_and_pub_cm_data(self):
        # Log current time.
        self.msg.header.stamp = rospy.Time.now()
        # Get CM data.
        for motor_idx in range(1, 7):
            self.msg.position[motor_idx-1] = Board.getBusServoPulse(motor_idx)  # Position
        # Publish the data.
        #self.monitor_pos_pub.publish(self.msg)


class ControlMotor:
    def __init__(self, node):
        self.node = node

        # Prepare initial values of the msg.
        self.msg = RosJointState()
        self.msg.name = ['Target value']
        self.msg.header.frame_id = 'not_relervant'
        self.msg.position = [0, 0]
        self.msg.temperature = [0]
        self.msg.voltage = [0]

        self.monitor_pos_pub = rospy.Publisher('/position_monitoring', RosJointState, queue_size=1)

    def send_and_pub_control_signal(self, target_value, duration, monitored_motor):
        # Set target value.
        Board.setBusServoPulse(monitored_motor, target_value, duration)
        # Publish the target value.
        #self.msg.header.stamp = rospy.Time.now()
        #self.msg.position[0] = target_value
        #self.msg.position[1] = duration
        #self.monitor_pos_pub.publish(self.msg)
        # Log the information.
        # rospy.loginfo('Publish control command: Position target: {}, Duration: {}ms'.format(target_value, duration))


def node_condition_monitoring(node, freq=100):
    cm_data_publisher = CMDataPublisher(node, freq)
    while not rospy.is_shutdown():
        cm_data_publisher.get_and_pub_cm_data()
        cm_data_publisher.r.sleep()


def node_control_robot(node, trajectories=[[500, 500, 500, 500, 500, 500]], durations_lists=[[1000, 1000, 1000, 1000, 1000, 1000]]):
    # Initialize ros node.
    robot_controller = ControlMotor(node)
    time.sleep(5)  # Sleep for 5 seconds. Time needed to start the listener on the PC side.

    # Loop over the trajectories.
    for trajectory, duration_list in zip(trajectories, durations_lists):
        # Loop over the motors.
        for monitored_motor in range(1, 7):
            motor_idx = monitored_motor - 1
            target_value = trajectory[motor_idx]
            duration = duration_list[motor_idx]
            # Set the target value.
            robot_controller.send_and_pub_control_signal(target_value, duration, monitored_motor)
            time.sleep(.0043)

        time.sleep(2)


if __name__ == '__main__':
    # Define trajectories.
    trajectories = [[500, 500, 500, 500, 500, 500], 
                    [90, 500, 80, 833, 620, 500],
                    [131, 500, 80, 833, 615, 939],
                    [99, 598, 72, 601, 304, 942],
                    [436, 595, 104, 603, 308, 939],
                    [436, 504, 103, 543, 463, 507],
                    [436, 504, 103, 636, 350, 506],
                    [27, 504, 104, 636, 349, 507]]
    durations_lists = [[1000, 1000, 1000, 1000, 1000, 1000], 
                       [1000, 1000, 1000, 1000, 1000, 1000],
                       [1000, 1000, 1000, 1000, 1000, 1000],
                       [1000, 1000, 1000, 1000, 1000, 1000],
                       [1000, 1000, 1000, 1000, 1000, 1000], 
                       [1000, 1000, 1000, 1000, 1000, 1000],
                       [1000, 1000, 1000, 1000, 1000, 1000],
                       [1000, 1000, 1000, 1000, 1000, 1000]]

    try:
        # Initialize ROS node in the main thread
        rospy.init_node('node_test_motor_position', anonymous=True)

        # Create two threads
        monitoring_freq = 1
        thread1 = threading.Thread(target=node_condition_monitoring, args=(rospy, monitoring_freq))
        thread2 = threading.Thread(target=node_control_robot, args=(rospy, trajectories, durations_lists))

        # Start the threads
        thread1.start()
        thread2.start()

        # Wait for both threads to finish
        thread1.join()
        thread2.join()

    except rospy.ROSInterruptException:
        pass
