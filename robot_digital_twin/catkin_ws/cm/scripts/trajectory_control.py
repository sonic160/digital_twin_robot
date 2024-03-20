#!/usr/bin/env python3
# Software License Agreement (BSD License)
# 
# This script controls a motor to turn following a unit-pulse signal. And monitor the response.


import threading, rospy, Board, time
from cm.msg import msg_cm as RosJointState
# import argparse


class CMDataPublisher:
    def __init__(self, node, io_block_flag: list, freq=10):
        self.node = node
        rate = self.node.get_param('~rate', freq)
        self.r = rospy.Rate(rate)
        self.io_block_flag = io_block_flag

        self.msg = RosJointState()
        self.msg.name = ['motor position']
        self.msg.header.frame_id = 'not_relervant'
        self.msg.position = [0, 0, 0, 0, 0, 0]
        self.msg.temperature = [0, 0, 0, 0, 0, 0]
        self.msg.voltage = [0, 0, 0, 0, 0, 0]

        # Start publisher
        self.monitor_pos_pub = rospy.Publisher('/condition_monitoring', RosJointState, queue_size=50)
        rospy.loginfo("Monitoring the position of motors" + " at " + str(rate) + "Hz")


    def safe_read_position(self, monitored_motor: int):
        ''' Read the position safely. This function verifies if the IO is occupied before perform the reading operation.
        If IO is blocked, it will waits until it is released. During the reading operation, it will block the IO.        
        '''
        # Check if the io is blocked:
        while self.io_block_flag[0]:
            print('Thread_CM: Waiting for the IO to be released!')
            pass

        # Block the IO and perform the reading action.
        self.io_block_flag[0] = True
        # Read the position.
        self.msg.position[monitored_motor-1] = Board.getBusServoPulse(monitored_motor)  # Position
        # Release the IO
        self.io_block_flag[0] = False


    def get_and_pub_cm_data(self):             
        # Log current time.
        self.msg.header.stamp = rospy.Time.now()
        # Get CM data.
        for monitored_motor in range(1, 7):
            self.safe_read_position(monitored_motor)
        # Publish the data.
        self.monitor_pos_pub.publish(self.msg)        


class ControlMotor:
    def __init__(self, node, io_block_flag: list):
        self.node = node
        self.io_block_flag = io_block_flag

        # Prepare initial values of the msg.
        self.msg = RosJointState()
        self.msg.name = ['Target value']
        self.msg.header.frame_id = 'not_relervant'
        self.msg.position = [0, 0, 0, 0, 0, 0]
        self.msg.temperature = [0, 0, 0, 0, 0, 0]
        self.msg.voltage = [0]

        self.monitor_pos_pub = rospy.Publisher('/position_monitoring', RosJointState, queue_size=1)


    def safe_control_motor(self, target_value: int, duration: int, monitored_motor: int):
        ''' Send the control command to a given motor safely. It verifies the IO is not occupied before sending the control command.
        During the sending operation, it will block the IO.
        '''
        # Check if the io is blocked:
        while self.io_block_flag[0]:
            print('Thread_Control: Waiting for the IO to be released!')
            pass

        # Block the IO and perform the reading action.
        self.io_block_flag[0] = True
        # Set target value.
        Board.setBusServoPulse(monitored_motor, target_value, duration)
        # Release the IO
        self.io_block_flag[0] = False


    def send_and_pub_control_signal(self, trajectory: list, duration_list: list):       
        # Log the current time.
        self.msg.header.stamp = rospy.Time.now()
        
        # Loop over the motors.
        for monitored_motor in range(1, 7):
            motor_idx = monitored_motor - 1
            target_value = trajectory[motor_idx]
            duration = duration_list[motor_idx]
            self.safe_control_motor(target_value, duration, monitored_motor)            
        # Sleep for 2 seconds. The time needed for the robot to finish one trajectory.
        time.sleep(2)
                
        # Publish the control command per trajectory.        
        self.msg.position = trajectory
        self.msg.temperature = duration_list
        self.monitor_pos_pub.publish(self.msg)
        # Log the information.
        rospy.loginfo('Publish control command: Position target: {}, Duration: {}ms'.format(self.msg.position, self.msg.temperature))       


def node_condition_monitoring(node, io_block_flag, freq=100):
    cm_data_publisher = CMDataPublisher(node, io_block_flag, freq)
    while not rospy.is_shutdown():
        cm_data_publisher.get_and_pub_cm_data()
        cm_data_publisher.r.sleep()


def node_control_robot(node, io_block_flag: list, trajectories=[[500, 500, 500, 500, 500, 500]], durations_lists=[[1000, 1000, 1000, 1000, 1000, 1000]]):
    # Initialize ros node.
    robot_controller = ControlMotor(node, io_block_flag)
    # Sleep for 5 seconds. Time needed to start the listener on the PC side.
    time.sleep(5)    
    # Loop over the trajectories. Send the control signals.
    for trajectory, duration_list in zip(trajectories, durations_lists):
        robot_controller.send_and_pub_control_signal(trajectory, duration_list)
  

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
    
    # Define the io block flag.
    io_block_flag = [False]

    try:
        # Initialize ROS node in the main thread
        rospy.init_node('node_test_motor_position', anonymous=True)

        # Create two threads
        monitoring_freq = 10
        thread1 = threading.Thread(target=node_condition_monitoring, args=(rospy, io_block_flag, monitoring_freq))
        thread2 = threading.Thread(target=node_control_robot, args=(rospy, io_block_flag, trajectories, durations_lists))

        # Start the threads
        thread1.start()
        thread2.start()

        # Wait for both threads to finish
        thread1.join()
        thread2.join()

    except rospy.ROSInterruptException:
        pass
