joint1_damping = 0;
joint2_damping = 0;
damp_pince = 1000; % damping coefficient for joints of the pince

mdl = "robot_model";

load_system(mdl)

ik = simscape.multibody.KinematicsSolver(mdl);

base = "robot_model/World/W";
follower = "robot_model/gripper_base/F";
addFrameVariables(ik,"gripper_base","translation",base,follower);
addFrameVariables(ik,"gripper_base","rotation",base,follower);

targetIDs = ["j1.Rz.q";"j2.Rz.q";"j3.Rz.q";"j4.Rz.q";"j5.Rz.q"] ;
addTargetVariables(ik,targetIDs);
outputIDs =["gripper_base.Translation.x";"gripper_base.Translation.y";...
    "gripper_base.Translation.z"];
addOutputVariables(ik,outputIDs);





joint1_damping = 0;
joint2_damping = 0;
damp_pince = 1000; % damping coefficient for joints of the pince

mdl = "robot_model";

load_system(mdl)

ik = simscape.multibody.KinematicsSolver(mdl);

base = "robot_model/World/W";
follower = "robot_model/gripper_base/F";
addFrameVariables(ik,"gripper_base","translation",base,follower);
addFrameVariables(ik,"gripper_base","rotation",base,follower);

targetIDs = ["j1.Rz.q";"j2.Rz.q";"j3.Rz.q";"j4.Rz.q";"j5.Rz.q"] ;
addTargetVariables(ik,targetIDs);
outputIDs =["gripper_base.Translation.x";"gripper_base.Translation.y";...
    "gripper_base.Translation.z"];
addOutputVariables(ik,outputIDs);






x = zeros(1000,1);
y = zeros(1000,1);
z = zeros(1000,1);
T = 10; % period
spline = zeros(1000,5);
% range for end effector space is aprroximately [-0.3,0.3] [-0.3,0.3] [0,0.3]
% 
% here the trajectory we choose for the end effector is 
% x = 0.1*cos( t * 2*pi/T  ) y = 0.1*sin( t * 2*pi/T) z = 0.15 + 0.1 * t/10


for i = 1:1000
    targets = [j1(i),j2(i),j3(i),j4(i),j5(i)];

    spline(i,:) = targets;

    [outputVec,statusFlag] = solve(ik,targets);
    x(i,1) = outputVec(1);
    y(i,1) = outputVec(2);
    z(i,1) = outputVec(3);

    
end


% Define the reference values generated with your formula
reference_x = zeros(1000, 1);
reference_y = zeros(1000, 1);
reference_z = zeros(1000, 1);

for i = 1:1000
    reference_x(i, 1) = 0.1 * cos(i / 100 * (2 * pi / T));
    reference_y(i, 1) = 0.1 * sin(i / 100 * (2 * pi / T));
    reference_z(i, 1) = 0.15 + 0.1 * (i / 100 / T);
end

% Calculate the absolute error between the calculated values and reference values
error_x = abs(x - reference_x);
error_y = abs(y - reference_y);
error_z = abs(z - reference_z);

% Now you have error_x, error_y, and error_z, which represent the absolute errors
% between your calculated values and the reference values for x, y, and z.

disp(mean(error_x));
disp(mean(error_y));
disp(mean(error_z));

% Your existing code to calculate error vectors and reference values

% Create a time vector (assuming your data is sampled every 0.01 seconds)
% Your existing code to calculate error vectors and reference values

% Create a time vector (assuming your data is sampled every 0.01 seconds)
% Your existing code to calculate error vectors and reference values

% Create a time vector (assuming your data is sampled every 0.01 seconds)
time = (0:0.01:9.99)';

% Create separate figures for each plot
figure;

% Plot the command input and X output
subplot(2, 1, 1);
plot(time, reference_x, 'b-');
title('Command Input (j1)');
xlabel('Time');
ylabel('Joint Angle');

subplot(2, 1, 2);
plot(time, x, 'r-');
title('X Output');
xlabel('Time');
ylabel('Position');

% Create a new figure
figure;

% Plot the command input and Y output
subplot(2, 1, 1);
plot(time, reference_y, 'b-');
title('Command Input (j2)');
xlabel('Time');
ylabel('Joint Angle');

subplot(2, 1, 2);
plot(time, y, 'g-');
title('Y Output');
xlabel('Time');
ylabel('Position');

% Create a new figure
figure;

% Plot the command input and Z output
subplot(2, 1, 1);
plot(time, reference_z, 'b-');
title('Command Input (j3)');
xlabel('Time');
ylabel('Joint Angle');

subplot(2, 1, 2);
plot(time, z, 'm-');
title('Z Output');
xlabel('Time');
ylabel('Position');



x = zeros(1000,1);
y = zeros(1000,1);
z = zeros(1000,1);
T = 10; % period
spline = zeros(1000,5);
% range for end effector space is aprroximately [-0.3,0.3] [-0.3,0.3] [0,0.3]
% 
% here the trajectory we choose for the end effector is 
% x = 0.1*cos( t * 2*pi/T  ) y = 0.1*sin( t * 2*pi/T) z = 0.15 + 0.1 * t/10


for i = 1:1000
    targets = [j1(i),j2(i),j3(i),j4(i),j5(i)];

    spline(i,:) = targets;

    [outputVec,statusFlag] = solve(ik,targets);
    x(i,1) = outputVec(1);
    y(i,1) = outputVec(2);
    z(i,1) = outputVec(3);

    
end


% Define the reference values generated with your formula
reference_x = zeros(1000, 1);
reference_y = zeros(1000, 1);
reference_z = zeros(1000, 1);

for i = 1:1000
    reference_x(i, 1) = 0.1 * cos(i / 100 * (2 * pi / T));
    reference_y(i, 1) = 0.1 * sin(i / 100 * (2 * pi / T));
    reference_z(i, 1) = 0.15 + 0.1 * (i / 100 / T);
end

% Calculate the absolute error between the calculated values and reference values
error_x = abs(x - reference_x);
error_y = abs(y - reference_y);
error_z = abs(z - reference_z);

% Now you have error_x, error_y, and error_z, which represent the absolute errors
% between your calculated values and the reference values for x, y, and z.

disp(mean(error_x));
disp(mean(error_y));
disp(mean(error_z));

% Your existing code to calculate error vectors and reference values

% Create a time vector (assuming your data is sampled every 0.01 seconds)
% Your existing code to calculate error vectors and reference values

% Create a time vector (assuming your data is sampled every 0.01 seconds)
% Your existing code to calculate error vectors and reference values

% Create a time vector (assuming your data is sampled every 0.01 seconds)
time = (0:0.01:9.99)';

% Create separate figures for each plot
figure;

% Plot the command input and X output
subplot(2, 1, 1);
plot(time, reference_x, 'b-');
title('Command Input (j1)');
xlabel('Time');
ylabel('Joint Angle');

subplot(2, 1, 2);
plot(time, x, 'r-');
title('X Output');
xlabel('Time');
ylabel('Position');

% Create a new figure
figure;

% Plot the command input and Y output
subplot(2, 1, 1);
plot(time, reference_y, 'b-');
title('Command Input (j2)');
xlabel('Time');
ylabel('Joint Angle');

subplot(2, 1, 2);
plot(time, y, 'g-');
title('Y Output');
xlabel('Time');
ylabel('Position');

% Create a new figure
figure;

% Plot the command input and Z output
subplot(2, 1, 1);
plot(time, reference_z, 'b-');
title('Command Input (j3)');
xlabel('Time');
ylabel('Joint Angle');

subplot(2, 1, 2);
plot(time, z, 'm-');
title('Z Output');
xlabel('Time');
ylabel('Position');


