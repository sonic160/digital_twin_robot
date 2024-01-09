#!/usr/bin/env python
import rospy
# Might need to change to cm.msg, depending on the package name.
from condition_monitoring.msg import msg_cm as RosJointState
import pandas as pd
import matplotlib.pyplot as plt


class DataCollector:
    def __init__(self):
        self.data = []

    def callback(self, msg):
        # Assuming 'position' is a field in the message
        position_data = msg.position[0]

        # Assuming 'stamp' is a field in the message header containing the timestamp
        timestamp = msg.header.stamp.to_sec()

        if msg.name[0] == 'Target value': # If it is the cmd signal
            # Add data to the list
            self.data.append({'timestamp': timestamp, 'position': position_data, 'command': True})
        else:
            self.data.append({'timestamp': timestamp, 'position': position_data, 'command': False})    

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

    def save_data_to_dataframe(self):
        if self.data:
            df = pd.DataFrame(self.data)

            # Post-processing
            df['time_since_start'] = (df['timestamp']-df['timestamp'][0])%1e3
            df['position_degree'] = df['position']/1000*240

            df.to_csv('/home/zhiguo/github_repo/digital_twin_robot/reference_model_development/condition_monitoring_demo/cm/scripts/collected_data.csv', index=False)
            print("Data saved to 'collected_data.csv'")

            # # Visualize the data.
            # plt.figure()
            # plt.plot(df['time_since_start'], df['position_degree'])
            # plt.xlabel('time since start (seconds)')
            # plt.ylabel('position (degrees)')
            # plt.grid(True)
            # plt.show()
            # plt.savefig('position.png')            


if __name__ == '__main__':
    data_collector = DataCollector()
    data_collector.run()
    data_collector.save_data_to_dataframe()
