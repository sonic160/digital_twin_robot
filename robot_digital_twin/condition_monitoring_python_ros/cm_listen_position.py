#!/usr/bin/env python
import rospy
# Might need to change to cm.msg, depending on the package name.
from condition_monitoring.msg import msg_cm as RosJointState
import pandas as pd
# import matplotlib.pyplot as plt
import os
from datetime import datetime


class DataCollector:
    ''' ### Description
    Class for collecting data from a contion-monitoring Ros topic.
    The test is to control a single motor to turn, in order to test its control performance.

    ### Initialization
    - base_path: Specify the path to the condition_monitoring_python_ros file.
    '''
    def __init__(self, base_path):
        self.data = []
        self.base_path = base_path
        self.motor = ''

    def callback(self, msg):
        # Assuming 'position' is a field in the message
        position_data = msg.position[0]

        # Assuming 'stamp' is a field in the message header containing the timestamp
        timestamp = msg.header.stamp.to_sec()

        if msg.name[0] == 'Target value': # If it is the cmd signal
            # Add data to the list
            self.data.append({'timestamp': timestamp, 'position': position_data, 'command': msg.position[1]})
        else:
            self.motor = msg.name[0]
            self.data.append({'timestamp': timestamp, 'position': position_data, 'command': 0})    

    def run(self):
        rospy.init_node('data_collector_node', anonymous=True)
        
        # Subscribe to the condition-monitoring and position-monitoring topics.
        sub_cm = rospy.Subscriber('condition_monitoring', RosJointState, self.callback)
        sub_pos_cmd = rospy.Subscriber('position_monitoring', RosJointState, self.callback)

        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Shutting down data collector node.")
            self.save_data_to_dataframe()
    

    def generate_data_path(self):
        ''' ### Description
        This function generate the path for saving the data under the base_path.
        It also generate the file names by logging the date and time for the current moment.

        ### Return
        - subfolder_path: Path for saving the data.        
        '''        
        base_path = self.base_path
        # Check if the folder exists, if not, create it
        folder_name = base_path + 'motor_position_test_data/'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # Create a subfolder with the current time as the folder name
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_name = current_time + '_' + self.motor + '.csv'
        
        return folder_name, file_name
    

    def save_data_to_dataframe(self):
        ''' ### Description
        Save the collected data.
        '''
        if self.data:
            df = pd.DataFrame(self.data)

            # Generate the folder to save the data.
            path, file_name = self.generate_data_path()

            # Post-processing
            df['time_since_start'] = (df['timestamp']-df['timestamp'][0])%1e3
            df['position_degree'] = df['position']/1000*240

            df.to_csv(path+'/'+file_name, index=False)
            print(f"Data saved to {file_name}")

            # # Visualize the data.
            # plt.figure()
            # plt.plot(df['time_since_start'], df['position_degree'])
            # plt.xlabel('time since start (seconds)')
            # plt.ylabel('position (degrees)')
            # plt.grid(True)
            # plt.show()
            # plt.savefig('position.png')            


if __name__ == '__main__':
    # Specify the base path for saving the data.
    base_path = '/home/zhiguo/github_repo/digital_twin_robot/robot_digital_twin/condition_monitoring_python_ros/'
    data_collector = DataCollector(base_path=base_path)
    data_collector.run()
    data_collector.save_data_to_dataframe()
