This document summarizes the development roadmap in the pole project IA (2023-2024): Digital twin for diagnosis. This project intends to develop a piloting use case of applying digital twin to simulate failure behavior and train AI-based fault diagnosis algorithms. The use cases will be implemented on a robot from LGI, Centralesupelec. The projects comprise of four work packages.

Note: This looks very ambitious. But don't feel stressed. You don't have to do it all by yourself. Some master students are working on the same project as well, and they will do part of the work. You only need to focus on the tasks assigned to you each week. This roadmap is to give you an overall picture of the whole project.

# WP1: Simscape model development

In this WP, we aim to develop a multi-body simulation model for our robot in Simscape. The multi-body simulation model should include the controller of each joint and forward and inverse kinematics simulation blocks. The simulation model will be used in the subsequent WPs to generate simulated data for training the fault diagnosis model.

- Task 1: Import the existing urdf file and create a multi-body model in Simscape.
  1. Follow this [tutorial](https://fr.mathworks.com/help/sm/ug/urdf-import.html#bvmwhdm-1).
  2. The urdf file is located in the folder "urdf".
  3. Deliverable of this task:
     - The model.
     - A simulation use case of the model. Show its performance in the nominal case, without any control. The robot should move randomly in this case.

- Task 2: Develop control blocks for the robot:
  1. Add pid controllers to each one of the six joints (See the [tutorial](https://www.youtube.com/watch?v=pDiwAA1cnb0!).
  2. Develop typical use cases to test each pid controller.
  3. Deliverable of this task:
     - The updated model with pid controller.
     - A summary report of the test cases, including the test conditions and the outputs.

- Task 3: Forward and inverse kinematics block:
  1. Follow the tutorial [here](https://fr.mathworks.com/help/sm/ref/simscape.multibody.kinematicssolver.html).
  2. Develop typical use cases to test the forward and inverse kinematic blocks.
     - Forward kinematics: Define each motor's rotational angles and simulate the end-effector's movement.
     - Inverse kinematics: Define a trajectory and calculate the required rotation degrees on the motor levels. Then, validate if the required trajectory can be fulfilled or not.
  4. Deliverable of this task:
     - The updated model with forward and inverse kinematic blocks.
     - A summary report of the test cases, including the test conditions and the outputs.

# WP2: Define typical failure scenarios and generate simulated failure data

In this WP, we aim to generate failure data from the digital twin model under different failure scenarios. The generated data will be used in the next WP to train AI-based fault diagnosis algorithms.

- Task 1: Identify the main failure modes of motors and how to model them in the multi-body model.
  1. Check from the literature what kind of failure could happen to the model physically.
  2. Develop simulation models for the identified failure modes.
     
- Task 2: Generate the training data for the fault diagnosis algorithm.

- Deliverable of this WP:
  1. A report summarizing the critical failure modes considered how to generate the simulation data.
  2. The generated dataset.
  3. A document describing the generated dataset.

# WP3: Develop AI-based fault diagnosis algorithms.

In this WP, the aim is to develop and test the performance of AI-based diagnosis algorithm.

- Deliverables:
  1. The AI-based algorithm for fault diagnosis.
  2. A notebook summarize all the model tried and their performance.
     - You need to use cross-validation to evaluate and compare the performance of different algorithms.
    
# WP4: Model application on real systems

In this WP, the objective is to test the developed algorithm in the physical system, i.e., the robot.

- Task 1: Condition-monitoring of the robot:
  1. Design a ROS node to broadcast the position, temperature, and voltage of each motor at a given frequency.
  2. Design a ROS node in matlab to receive the message.
  3. Design a GUI (dashboard) to visualize in realtime the collected condition-monitoring data.
- Task 2: Implement the fault diagnosis algorithm based on the collected condition-monitoring data.





  
