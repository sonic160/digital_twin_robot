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
guesses = [0,0,0,0,0];
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
    targets = [0+0.1*cos(i/100*(2*pi/T)),0+0.1*sin(i/100*(2*pi/T)),0.15+0.1*(i/100/T)];
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

