The condition-monitoring program in the robot is sending out condition-monitoring data as a customized Ros message. The message is defined in ```robot_digital_twin\catkin_ws\cm\msg\msg_cm.msg```. If we want to read the message in Matlab, we need to build this customized message in Matlab first. 

Before starting, please make sure that:
1. You have Python installed in your computer. Matlab Ros Toolbox support versions **3.8 - 3.10**. Please make sure you installed a supported version. Please install the interpreter directly from the [official website](https://www.python.org/downloads/), not from Anaconda.
2. Matlab requires Visual Studio 2019 or later to build the message. Follow the instruction below:
    - Install Visual Studio C++ 2019 or later. Download the community version from [here](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community&channel=Release&version=VS2022&source=VSLandingPage&cid=2030&passive=false)
    - Make sure that the "Desktop development with C++" workload is selected when installing Visual Studio (see the tutorial [here](https://learn.microsoft.com/en-us/visualstudio/install/install-visual-studio?view=vs-2022#step-4---choose-workloads)).

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
<image src=screen_shots/build_msg_matlab.png width=600></image>

    - If you see an error message related to Matlab cannot find Python installation, you should follow the following steps:
        - Make sure you have Python installed on your computer. Please note that Matlab supports Python **3.8 - 3.10**.
        - Follow the instruction given in the error message, provide the necessary python path. If you are not sure what is the path of your python installation, you can open command line window, and use the commond "where python".
      ![Alt text](screen_shots/where_python.png)

    - If you see an error message: `Current compiler MinGW64 Compiler (C++) is not supported for ROS build. To choose a compiler, run 'mex -setup cpp'.` 
        - Install Visual Studio C++ 2019 or later. Download the community version from [here](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community&channel=Release&version=VS2022&source=VSLandingPage&cid=2030&passive=false)
        - Make sure that the "Desktop development with C++" workload is selected when installing Visual Studio (see the tutorial [here](https://learn.microsoft.com/en-us/visualstudio/install/install-visual-studio?view=vs-2022#step-4---choose-workloads)).

6. Follow the instructions you got from the screen above, which is summarized in `after_generate_msg.m`. If everything goes well, you should be able to find `msg_cm` appearing in the message list.

7. Normally, the message needs to be complied only once. However, if Matlab fails to find the `msg_cm` message, you can try:
    - Rerun step 5.
    - If your path name is too long, rerun Step 4, and then step 5.
    - If it does not work, recompile the message: Start from Step 1.

