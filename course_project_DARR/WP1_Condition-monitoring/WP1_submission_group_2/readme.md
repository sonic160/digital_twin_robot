## Context

**Course** : Data Analysis for Risk and Reliability
**Project** : Digital Twin Robot
**Members of the group** : Antonis Kostopoulos, Firas Ben Nasr, Thomas Watteau
**Jalon** : WP1 - Collect, save and plot the position, temperature and voltage of the 6 motors.


## What changed in the updated app prototype

- **Choice of the motor** : Previously, only the data from the 1st motor was saved and plotted. Now, the data from the 6 motors is saved in CSV files, and using a DropDown menu, the user can choose the motor from which the data is plotted.
- **Graphs** : Previously, the data plotted in the 3 graphs was hardly readable. This is why we changed the colors and y-limits.
- **Gauges** : We wanted an immediate view on the data, even though it fluctuates a lot. That's why we implemented a gauge to see the current value of the position, temperature and voltage, using the same y-limits as the graphs.


## How to use the updated app

Start your robot so that it broadcasts its data. If you don't have the robot, you can create a fake robot that broadcasts data : execute the file "test_offline_start_fake_robot.m".
Get the IP address (from your real or fake robot), open the file "app_prototype.mlapp" and change the previous IP address at line 192 : `rosinit('10.152.100.6', 11311)`.
Save the file and execute it.
Before clicking Start, you can choose which motor you would like to plot the data from, using the DropDown menu at the top of the screen.