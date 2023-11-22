The application *app_prototype_fault_detection_master.mlapp* shows a prototype of data collection application with the functionality of injecting failures. The main steps of the program are as follows:
1. Connect to the robot and read condition-monitoring data through defining a subscriber.
2. Extract the condition-monitoring data from the message from ROS.
3. Visualize in realtime the position, temperature and voltage.
4. Inject a fault into the robot by changing the temperature.
5. Label the period when the failure is injected as 1, otherwise, label 0.
6. Save the collected data in a .csv file.

To test it, you should follow the following steps:
1. Start the condition-monitoring program in the robot. To do this, you need:
    1. Connect a PC to the robot through nomachine. See the tutorial [here](/Ref_NoMachine%20Installation%20and%20Connection.pdf).
    2. Open a terminal for the robot through nomachine.
    3. In the terminal, type the following command to start the condition-monitoring program in the robot:
        - `source catkin_ws/devel/setup.bash`
        - `rosrun condition_monitoring condition_monitoring.py`
        - If successful, you should see from the nomachine that the robot is sending out condition-monitoring data.
2. Change the ip address in the rosinit function to the ip address of the actual or fake robot.
3. Click the *start* button to start data collection and data visualization. 
4. In the drop-down, select the motor you want to inject failure. Then, click the faiulre injection button. Inject the failure physically. When the failure injection is over, click the failure injection button again to make it "off".
5. Click the *stop* button to save the collected data.
