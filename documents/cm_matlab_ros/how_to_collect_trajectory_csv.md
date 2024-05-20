In this document, you will find full instructions on how to obtain on a given PC .csv files containing trajectory, command and command duration data.

## Preparations

-Prepare PC

In order to ensure consistent versions of ROS between the robot arm's controller and the PC, we reccomend you install ROS Melodic. If your OS does not support ROS melodic, you can setup and use a Docker image to contain it.  The following links can serve as reference for how to build this :
	'-https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository'
	'-http://wiki.ros.org/docker/Tutorials/Docker (please specify pull ros:melodic)'
	
You should then clone locally the dtr_robot_digital_shaddow (https://github.com/sonic160/dtr_robot_digital_shaddow) repository, in order to provide the appropriate folder in which to mount the ROS docker image.

You must then, in order :

	-proceed with the mount, with the command 'sudo docker run -it -v /home/<location of your repo clone>/dtr_robot_digital_shaddow/pc_side/catkin_ws:/root/catkin_ws ros:melodic'
	
	-build the custom messages, by running 'cd root/catkin_ws/' and 'source devel/setup.bash '.
	-if you are working from a docker image, it is likely that you will be lacking some critical python imports for the code to run. The following are a sample of the series of commands you may have to run - once pip is installed, you should be able to download any other missing libraries in this fashion.


		'apt-get update'
		'apt-get install -y ros-melodic-rospy'
		'apt-get install -y python3-pip'
		'pip3 install pandas'
		'pip3 install pyyaml'
		'pip3 install rospkg'
		
	-the PC side is now ready.
	


-Prepare Robot

The robot should come with ROS melodic installed. Therefore, you can directly proceed to clone the repository, and run 'cd root/catkin_ws/' and 'source devel/setup.bash ', located in the catkin_ws folder to build the messages. If any python libraries are missing, repeat the steps described above.

## Usage

-Running the simulation and collecting the data

You should now be able to run, on the robot side, the python files under 'src/cm/scripts'; for instance execute 'python3 failure_simulation.py'. This should start robot mouvement shortly. After doing this, you should launch on the PC side, under 'pc_side/catkin_ws/src/cm_listener', the command 'python3 cm_listen_trajectory.py'. An empty output is to be expected. After the robot has stopped moving, hitting Ctrl+c on the PC side will save the recorded data as .csv files, under the directory 'catkin_ws/src/cm_listener/data_repository'.

## Exploiting the csv files

-Three files should have been generated, containing : 
	- a series of mesaurements at regular time intervals ('trajectory_monitoring_position.csv')
	- a series of commanded positons ('trajecory_monitoring_cmd.csv')
	- a series of times, in which the associated commands where to be executed (('trajecory_monitoring_cmd_duration.csv')
	
	
- TODO : finish Matlab-side csv reader that converts the csv data into a format consistent with the trained AI in order to perform state predictions.


	
	
	

