In this document, you will find a demo of how to create a fake robot in Matlab so that you can test your program offline. To do so, you just need to run `robot_digital_twin\condition_monitoring_matlab_ros\test_application_without_robot\test_offline_start_fake_robot.m`

If everything goes well, you should see a command line window poping up, like the following:
![Alt text](screen_shots/rosmaster.png)

And the Matlab command window will like this:
![Alt text](screen_shots/matlab_cmd_window.png)

Note that the 192.168.1.14 is the ip address of the generated robot. **You will need to provide this address to the "rosinit" command in your client program.** This generated robot will mimic the behavior of the real robot, i.e., publish condition-monitoring data (fake one) in the topic "/condition-monitoring" at a rate of $10$ Hz. 

It could be possible that during your first run, Matlab will ask you to specify dictionary of your python installation. You just need to follow the instruction and provide the necessary python path. If you are not sure what is the path of your python installation, you can open command line window, and use the commond "where python".
![Alt text](screen_shots/where_python.png)

Once the fake robot is created, you need to keep this matlab terminal open, and open another matlab terminal (simply click again the Matlab shortcut) as client to test the condition-monitoring application (follow this [tutorial](how_to_use_condition_monitoring_app.md)). 

Please note that the fake robot is just sending some random monitoring data.

When you finish testing your program, you need to end the fake robot manually: Go the first Matlab terminal, press "Ctrl + C" and then enter "rosshutdown" in Matlab command window.
