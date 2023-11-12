num_data_points = 4;
forward_x = zeros(num_data_points,1);
forward_y = zeros(num_data_points,1);
forward_z = zeros(num_data_points,1);
output_pos = zeros(num_data_points,3);
forward_xyz = [];

model_name = 'ifk_updated_armpi_fpv';
load_system(model_name);

T=10;
for data_point = 1:num_data_points
    targets_trajectory = zeros(1000,3);
    for i = 1:num_data_points
    targets_trajectory(i, :) = [0.1 * cos(i/100*(2*pi/T)), 0.1 * sin(i/100*(2*pi/T)), 0.15 + 0.1 * (i/100/T)];
    [joint1_ts, joint2_ts, joint3_ts, joint4_ts, joint5_ts] = InverseKinematic(targets_trajectory);
    simOut = sim(model_name);

    [x, y, z] = ForwardKinematic(j1, j2, j3, j4, j5);

    % Append the results to lists
    forward_x = [forward_x; x];
    forward_y = [forward_y; y];
    forward_z = [forward_z; z];
    current_iteration = [x, y, z];
    
    % Append the current_iteration to forward_xyz
    forward_xyz = [forward_xyz; current_iteration];
    
    end





end








function [joint1_ts, joint2_ts, joint3_ts, joint4_ts, joint5_ts] = InverseKinematic(targetlist)
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

targetIDs = ["gripper_base.Translation.x";"gripper_base.Translation.y";...
    "gripper_base.Translation.z"];
addTargetVariables(ik,targetIDs);
outputIDs =["j1.Rz.q";"j2.Rz.q";"j3.Rz.q";"j4.Rz.q";"j5.Rz.q"];
addOutputVariables(ik,outputIDs);

guessesIDs = ["j1.Rz.q";"j2.Rz.q";"j3.Rz.q";"j4.Rz.q";"j5.Rz.q"];
guesses = [3,3,3,3,3];
addInitialGuessVariables(ik,guessesIDs);


j1 = zeros(1000,1);
j2 = zeros(1000,1);
j3 = zeros(1000,1);
j4 = zeros(1000,1);
j5 = zeros(1000,1);
T = 10; % period
spline = zeros(1000,3);
% range for end effector space is aprroximately [-0.3,0.3] [-0.3,0.3] [0,0.3]
% here the trajectory we choose for the end effector is 
% x = 0.1*cos( t * 2*pi/T  ) y = 0.1*sin( t * 2*pi/T) z = 0.15 + 0.1 * t/10

for i = 1:1000
    %targets = targetlist(i);
    targets = [0.1 * cos(i/100*(2*pi/T)), 0.1 * sin(i/100*(2*pi/T)), 0.15 + 0.1 * (i/100/T)];
    spline(i,:) = targets;
    if i>1 
        guesses = [j1(i-1,1),j2(i-1,1),j3(i-1,1),j4(i-1,1),j5(i-1,1)];
    end
    [outputVec,statusFlag] = solve(ik,targets,guesses);
    j1(i,1) = outputVec(1);
    j2(i,1) = outputVec(2);
    j3(i,1) = outputVec(3);
    j4(i,1) = outputVec(4);
    j5(i,1) = outputVec(5);
    
end

joint1_ts = timeseries(j1/180*pi,0:0.01:9.99);
joint2_ts = timeseries(j2/180*pi,0:0.01:9.99);
joint3_ts = timeseries(j3/180*pi,0:0.01:9.99);
joint4_ts = timeseries(j4/180*pi,0:0.01:9.99);
joint5_ts = timeseries(j5/180*pi,0:0.01:9.99);

end

function [x, y ,z] = ForwardKinematic(j1, j2, j3, j4, j5);
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


for i = 1:1000
    targets = [j1(i),j2(i),j3(i),j4(i),j5(i)];

    spline(i,:) = targets;

    [outputVec,statusFlag] = solve(ik,targets);
    x(i,1) = outputVec(1);
    y(i,1) = outputVec(2);
    z(i,1) = outputVec(3);

    
end




end