#!/usr/bin/env python
import rospy
# Might need to change to cm.msg, depending on the package name.
from condition_monitoring.msg import msg_cm as RosJointState
import pandas as pd
import matplotlib.pyplot as plt


class DataCollector:
    def __init__(self):
        self.position_data = []
        self.command_position = []
        self.command_duration = []


    def position_monitoring(self, msg):
        # Get the position monitoring data.
        position = msg.position

        # Get the time stamp.
        timestamp = msg.header.stamp.to_sec()

        # Add the current message into a list.
        self.position_data.append({
            'timestamp': timestamp, 
            'motor_1': position[0],
            'motor_2': position[1],
            'motor_3': position[2],
            'motor_4': position[3],
            'motor_5': position[4],
            'motor_6': position[5]
        })


    def command_monitoring(self, msg):
        # Get the position monitoring data.
        trajectory = msg.position
        duration_list = msg.temperature

        # Get the time stamp.
        timestamp = msg.header.stamp.to_sec()

        # Add the current message into a list.
        self.command_position.append({
            'timestamp': timestamp, 
            'motor_1': trajectory[0],
            'motor_2': trajectory[1],
            'motor_3': trajectory[2],
            'motor_4': trajectory[3],
            'motor_5': trajectory[4],
            'motor_6': trajectory[5]
        })

        self.command_duration.append({
            'timestamp': timestamp, 
            'motor_1': duration_list[0],
            'motor_2': duration_list[1],
            'motor_3': duration_list[2],
            'motor_4': duration_list[3],
            'motor_5': duration_list[4],
            'motor_6': duration_list[5]
        })


    def run(self):
        rospy.init_node('data_collector_node', anonymous=True)
        
        # Subscribe to the condition-monitoring and position-monitoring topics.
        sub_cm = rospy.Subscriber('condition_monitoring', RosJointState, self.position_monitoring)
        sub_pos_cmd = rospy.Subscriber('position_monitoring', RosJointState, self.command_monitoring)

        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Shutting down data collector node.")
            self.save_data_to_dataframe()

    def save_data_to_dataframe(self):
        # Position monitoring data.
        if self.position_data:
            df = pd.DataFrame(self.position_data)

            # Post-processing
            df['time_since_start'] = (df['timestamp']-df['timestamp'][0])%1e3
            for col in df.columns[1:]:
                df[col] = df[col]/1000*240

            df.to_csv('/home/zhiguo/github_repo/digital_twin_robot/reference_model_development/condition_monitoring_demo/cm/scripts/trajectory_monitoring_position.csv', index=False)
            print("trajectory_monitoring_position.csv'")

        # Command data.
        if self.command_position:
            df_cmd = pd.DataFrame(self.command_position)
            # Post-processing
            df_cmd['time_since_start'] = (df_cmd['timestamp']-df['timestamp'][0])%1e3
            for col in df_cmd.columns[1:]:
                df_cmd[col] = df_cmd[col]/1000*240

            df_cmd.to_csv('/home/zhiguo/github_repo/digital_twin_robot/reference_model_development/condition_monitoring_demo/cm/scripts/trajectory_monitoring_cmd.csv', index=False)
            print("trajectory_monitoring_cmd.csv'")

        # Command duration.
        if self.command_duration:
            df_cmd_duration = pd.DataFrame(self.command_duration)
            df_cmd_duration.to_csv('/home/zhiguo/github_repo/digital_twin_robot/reference_model_development/condition_monitoring_demo/cm/scripts/trajectory_monitoring_cmd_duration.csv', index=False)
            print("trajectory_monitoring_cmd_duration.csv'")
         


if __name__ == '__main__':
    data_collector = DataCollector()
    data_collector.run()
    data_collector.save_data_to_dataframe()
