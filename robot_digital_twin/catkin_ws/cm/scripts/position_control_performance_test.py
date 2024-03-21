#!/usr/bin/env python3
# Software License Agreement (BSD License)
# 
# This script controls a motor to turn following a unit-pulse signal. And monitor the response.


import threading, rospy, Board, time
from cm.msg import msg_cm as RosJointState
import argparse


class CMDataPublisher:
    def __init__(self, node, freq=10, monitored_motor=6):
        self.node = node
        rate = self.node.get_param('~rate', freq)
        self.r = rospy.Rate(rate)

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

    def get_and_pub_cm_data(self):
        # Log current time.
        self.msg.header.stamp = rospy.Time.now()
        # Get CM data.
        motor_idx = self.monitored_motor
        self.msg.position[0] = Board.getBusServoPulse(motor_idx)  # Position
        # Publish the data.
        self.monitor_pos_pub.publish(self.msg)


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
        time.sleep(5)  # Sleep for 3 seconds.
        # Set target value.
        Board.setBusServoPulse(monitored_motor, target_value, duration)
        # Publish the target value.
        self.msg.header.stamp = rospy.Time.now()
        self.msg.position[0] = target_value
        self.msg.position[1] = duration
        self.monitor_pos_pub.publish(self.msg)
        # Log the information.
        rospy.loginfo('Publish control command: Position target: {}, Duration: {}ms'.format(target_value, duration))


def node_condition_monitoring(node, freq=100, monitored_motor=6):
    cm_data_publisher = CMDataPublisher(node, freq, monitored_motor)
    while not rospy.is_shutdown():
        cm_data_publisher.get_and_pub_cm_data()
        cm_data_publisher.r.sleep()


def node_control_robot(node, target_value=550, original_value=500, duration=50, monitored_motor=6):
    # Initialize ros node.
    robot_controller = ControlMotor(node)
    # First movement.
    robot_controller.send_and_pub_control_signal(target_value, duration, monitored_motor)
    # Going back.
    robot_controller.send_and_pub_control_signal(original_value, duration, monitored_motor)


if __name__ == '__main__':
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Input parameter values through ommand-line.')

    # Add arguments
    parser.add_argument('-target_value', '--target_value', type=int, default=550)
    parser.add_argument('-original_value', '--original_value', type=int, default=500)
    parser.add_argument('-duration', '--duration', type=int, default=50)
    parser.add_argument('-monitored_motor', '--monitored_motor', type=int, default=6)

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the values using args.
    target_value = args.target_value
    original_value = args.original_value
    duration = args.duration
    monitored_motor = args.monitored_motor

    try:
        # Initialize ROS node in the main thread
        rospy.init_node('node_test_motor_position', anonymous=True)

        # Create two threads
        monitoring_freq = 100
        thread1 = threading.Thread(target=node_condition_monitoring, args=(rospy, monitoring_freq, monitored_motor))
        thread2 = threading.Thread(target=node_control_robot, args=(rospy, target_value, original_value, duration, monitored_motor))

        # Start the threads
        thread1.start()
        thread2.start()

        # Wait for both threads to finish
        thread1.join()
        thread2.join()

    except rospy.ROSInterruptException:
        pass
