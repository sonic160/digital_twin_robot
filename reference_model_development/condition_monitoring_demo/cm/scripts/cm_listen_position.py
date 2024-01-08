#!/usr/bin/env python
import rospy
# Might need to change to cm.msg, depending on the package name.
from condition_monitoring.msg import msg_cm as RosJointState
import pandas as pd


class DataCollector:
    def __init__(self):
        self.data = []

    def callback(self, msg):
        # Assuming 'position' is a field in the message
        position_data = msg.position[0]

        # Assuming 'stamp' is a field in the message header containing the timestamp
        timestamp = msg.header.stamp.to_sec()

        # Add data to the list
        self.data.append({'timestamp': timestamp, 'position': position_data})

        if msg.name[0] == 'Target value':
            print(msg)

    def run(self):
        rospy.init_node('data_collector_node', anonymous=True)
        
        # Replace 'your_topic' with the actual topic you want to subscribe to
        s_1 = rospy.Subscriber('condition_monitoring', RosJointState, self.callback)
        s_2 = rospy.Subscriber('position_monitoring', RosJointState, self.callback)

        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Shutting down data collector node.")
            self.save_data_to_dataframe()

    def save_data_to_dataframe(self):
        if self.data:
            df = pd.DataFrame(self.data)
            df.to_csv('collected_data.csv', index=False)
            print("Data saved to 'collected_data.csv'")


if __name__ == '__main__':
    data_collector = DataCollector()
    data_collector.run()
    data_collector.save_data_to_dataframe()
