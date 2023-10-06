This document summarizes the development roadmap in the pole project IA (2023-2024): Digital twin for diagnosis. 

# WP1: Simscape model development
In this WP, we aim to develop a multi-body simulation model for our robot in simscape. The multi-body simulation model should include the controller of each joint and forward and inverse kinematics simulation blocks. The simulation model will be used in the subsequent WPs to generate simulated data for training the fault diagnosis model.

- Task 1: Import the existing urdf file and create a multi-body model in Simscape.
  1. Follow the tutorial here:
  2. The urdf file is located in the folder "urdf"
  3. Deliverable of this task:
     - The model.
     - A simulation use case of the model. Show its performance in the nominal case, without any control. The robot should move randomly in this case.

- Task 2: Develop control block for the robot:
  -- Add pid controllers to each one of the six joints.
  -- Develop typical use cases to test each one of the pid controller.
  -- Deliverable of this task:
     --- The updated model with pid controller.
     --- A summary report of the test cases: including the test conditions and the outputs.

- Task 2: Forward and inverse kinematics block:
  -- Follow the tutorial here:
  
