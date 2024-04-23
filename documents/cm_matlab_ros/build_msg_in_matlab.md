The condition-monitoring program in the robot is sending out condition-monitoring data as a customized Ros message. The message is defined in ```robot_digital_twin\catkin_ws\cm\msg\msg_cm.msg```. If we want to read the message in Matlab, we need to build this customized message in Matlab first. 

This can be done through the following steps:

1. Set the current dictionary of Matlab to be
<p align="center">
    <code>
        robot_digital_twin\condition_monitoring_matlab_ros
    </code>
</p>

2. Open `generate_msg.m` in the current dictionary. Replace `base_dictionary = 'Z:'` with the actual path to your `catkin_ws` folder. In this repository, the `catkin_ws` folder is located under `robot_digital_twin\catkin_ws`. You should add the path of your `robot_digital_twin` folder to this path as well. For example, if your `robot_digital_twin` folder is located under `C:\users\`, then the complete path should be
<p align="center">
    <code>
        C:\users\robot_digital_twin\catkin_ws
    </code>
</p>

3. Replace `folderpath = 'Z:\cm\'` with the actual path to your `cm` folder. If your `robot_digital_twin` folder is located under `C:\users\`, then the complete path should be
<p align="center">
    <code>
        C:\users\robot_digital_twin\catkin_ws\cm
    </code>
</p>

4. Sometimes, if your path name is too long, you might got an error when building the message like [this post](https://fr.mathworks.com/matlabcentral/answers/1571318-why-does-rosgenmsg-in-ros-toolbox-fail-when-working-in-a-directory-with-a-long-absolute-path-name). In this case, you can use `sust` command from Powershell:
    1. Open Powershell.
    2. Run `Z: "your long path name"`, where `Z:` is used to replace your long path name. Then, you can use `base_dictionary = 'Z:'` and `folderpath = 'Z:\cm\'` instead.

5. Run `generate_msg.m` in the current dictionary. You should be able to see the message built process start in the Matlab console. It might take a few minutes. If successful, you would see a message like this:
<image src=screen_shots/build_msg_matlab.png width=600>

6. Follow the instructions you got from the screen above, which is summarized in `after_generate_msg.m`. If everything goes well, you should be able to find `msg_cm` appearing in the message list.

7. Normally, the message needs to be complied only once. However, if Matlab fails to find the `msg_cm` message, you can try:
    - Rerun step 5.
    - If your path name is too long, rerun Step 4, and then step 5.
    - If it does not work, recompile the message: Start from Step 1.

