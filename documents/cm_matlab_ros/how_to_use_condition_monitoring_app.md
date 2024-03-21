In this document, you will find a tutorial on how to use the condition-monitoring application to collect the realtime condition-monitoring data from the robot. The condition-monitoring application is developed in Matlab application designer. It will collect the position, temperature and voltage of the six motors of the robot in realtime, and once the collection is stopped, the collected data will be saved in a .csv files.

## Preparations
- Prepare Matlab
    - This application needs Matlab R2023a or later.
    - We rely on a customized Ros message to reveive the condition-monitoring data from the robot. This message needs to be compiled and built first in Matlab. For this, please follow the tutorial [here](build_msg_in_matlab.md).    
- If you are using the real robot: 
    - We use [Hiwonder ArmPi FPV](https://www.hiwonder.com/products/armpi-fpv?variant=39341129203799). This is a robot based on Rapspebery Pi 4 and ROS 1 Melodic.
    - Follow the tutorial [here](prepare_robot.md) to prepare the robot, and start the condition-monitoring program on the robot.
- If you are testing withought the real robot:
    - You need to create a virtual robot in another Matlab instance, and send the simulated condition-monitoring data from there. Please follow [tutorial](create_a_fake_robot_for_testing.md).


## Start the Matlab application to collect the data

The data collection application is available from `robot_digital_twin\condition_monitoring_matlab_ros\matlab_application\data_collection_for_ref_model.mlapp`. With it, we can:
1. Connect to the robot and receive condition-monitoring data sent from the robot. This is done through defining a subscriber in Matlab and subscribe to the Ros topic "/condition_monitoring".
2. Extract the condition-monitoring data from the messages sent from the robot.
3. Visualize in realtime the position, temperature and voltage.
4. Save the collected data in .csv files .

To test it, you should follow the following steps:
1. Start the condition-monitoring program in the robot ([tutorial](prepare_robot.md)) or create a fake robot for testing ([tutorial](create_a_fake_robot_for_testing.md)).
2. Change the working dictionary of Matlab to the folder `robot_digital_twin\condition_monitoring_matlab_ros\matlab_application`.
2. Open `connect_and_monitor.m`. Change the ip address to the actual ip address of the robot.
3. Click the **Start condition monitoring** button to start data collection and data visualization. 
4. Click the **Stop condition monitoring** button to save the collected data.

## Expected results

The program will visualize the collected data in realtime:
<p align="center">
    <image src=screen_shots/demo_condition_monitoring.gif/>
</p>
In this demonstration above, for each motor, from top to bottom, we visualize the position, temperature and voltage. It can be seen that the robot is moving as the positions change from time to time, and the temperature and voltage are changing as they are affected by the robot movement.

Once the stop button is clicked, the application will generated a new folder under `robot_digital_twin\condition_monitoring_matlab_ros\matlab_application\collected_data`. The name of the folder is in the format of `YYYYMMDD_HHMMSS`, indicating the time of data collection. Under this folder, there will be six .csv files, one for each motor.
