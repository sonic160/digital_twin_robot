
%load('trajCS.mat')
%longeur de la time series

len_time_series=10000;
sample_time=10/len_time_series;
%trajectoire CS

x = trajCS.x;
y = trajCS.y;
z = trajCS.z;
scaleFactorX = 3;
scaleFactorY = 3;

% Scaling the vectors
x_scaled = x * scaleFactorX;
y_scaled = y * scaleFactorY;

% Plotting in 3D
% figure;
% plot3(x_scaled, y_scaled, z, 'o-');
% grid on;
% xlabel('X-axis');
% ylabel('Y-axis');
% zlabel('Z-axis');
% title('3D Plot Example');

% Desired number of points
desiredPoints = len_time_series;

% Create indices for downsampling
indices = round(linspace(1, length(x), desiredPoints));

% Downsample the vectors
x_downsampled = x_scaled(indices);
y_downsampled = y_scaled(indices);
z_downsampled = z(indices)+0.05;

% Plotting downsampled in 3D
figure;
plot3(x_downsampled, y_downsampled, z_downsampled, 'o-');
grid on;
xlabel('X-axis');
ylabel('Y-axis');
zlabel('Z-axis');
title('3D Plot downsampleed');

datapoints =[x_downsampled, y_downsampled,z_downsampled];

disp(size(datapoints))

%Prepwork
model_name = 'main3_armpi_fpv';
load_system(model_name);
joint1_damping = 0;
joint2_damping = 0;
damp_pince = 1000;
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

%simul length: len_time_series/100= legnth of simulation in seconds

j1 = zeros(len_time_series,1);
j2 = zeros(len_time_series,1);
j3 = zeros(len_time_series,1);
j4 = zeros(len_time_series,1);
j5 = zeros(len_time_series,1);
T = 10; % period
spline = zeros(len_time_series,3);
targets = zeros(len_time_series,3);


m0=[transpose(1:len_time_series), zeros(len_time_series, 1)];
m1=[transpose(1:len_time_series), ones(len_time_series, 1)];




 for t = 1:len_time_series
        datapoint =datapoints(t,:);
        %datapoint =datapoints(t,:)*0+[0.1*t, 0.1*t, 0.1*t];
        %datapoint = [0+k*0.1*cos(t/100*(2*pi/T)),0+k*0.1*sin(t/100*(2*pi/T)),0.15+k*0.1*(t/100/T)];
        spline(t,:)  = datapoint;
        targets(t,:) = datapoint;

        
        
        if t>1 
            guesses = [j1(t-1,1),j2(t-1,1),j3(t-1,1),j4(t-1,1),j5(t-1,1)];
        end
    

        [outputVec,statusFlag] = solve(ik,datapoint,guesses);
        j1(t,1) = outputVec(1);
        j2(t,1) = outputVec(2);
        j3(t,1) = outputVec(3);
        j4(t,1) = outputVec(4);
        j5(t,1) = outputVec(5);
end
        
end_time_value_in_seconds= (len_time_series-1)*0.001;

joint1_ts = timeseries(j1/180*pi,0:0.001:end_time_value_in_seconds);
joint2_ts = timeseries(j2/180*pi,0:0.001:end_time_value_in_seconds);
joint3_ts = timeseries(j3/180*pi,0:0.001:end_time_value_in_seconds);
joint4_ts = timeseries(j4/180*pi,0:0.001:end_time_value_in_seconds);
joint5_ts = timeseries(j5/180*pi,0:0.001:end_time_value_in_seconds);

for j=0:0   %il faut réparer les moteurs 4/5/6
    fprintf('Motor off is:%d\n',j);
    error1=m1;
    error2=m1;
    error3=m1;
    error4=m1;
    error5=m1;
    error6=m1;
   switch j
        case 1
            error1=m0;
        case 2
            error2=m0;
        case 3
            error3=m0;
        case 4
            error4=m0;
        case 5
            error5=m0;
        case 6
            error6=m0;
   end

    %on ajoute déjà les trajectoires cibles
    disp("----------------")
    disp("----------------")
    simOut = sim(model_name);
    disp("----------------")
    disp("----------------")
    

    j1o = simOut.j1.Data;
    j2o = simOut.j2.Data;
    j3o = simOut.j3.Data;
    j4o = simOut.j4.Data;
    j5o = simOut.j5.Data;
    j1o = j1o*180/pi;
    j2o = j2o*180/pi;
    j3o = j3o*180/pi;
    j4o = j4o*180/pi;
    j5o = j5o*180/pi;

    [x, y, z] = ForwardKinematic(j1o, j2o, j3o, j4o, j5o,len_time_series); 
    % Plotting simulated
    figure;
    plot3(x, y, z, 'o-');
    grid on;
    xlabel('X-axis');
    ylabel('Y-axis');
    zlabel('Z-axis');
    title('3D Plot downsampleed');

end


function [x, y ,z] = ForwardKinematic(j1, j2, j3, j4, j5,len_time_series)
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
    
    x = zeros(len_time_series,1);
    y = zeros(len_time_series,1);
    z = zeros(len_time_series,1);
    T = 10; % period
    %spline = zeros(len_time_series,5);
    
    len = size(j1);
    for i = 1:len_time_series
        targets = [j1(i),j2(i),j3(i),j4(i),j5(i)];
    
        [outputVec,statusFlag] = solve(ik,targets);
        x(i,1) = outputVec(1);
        y(i,1) = outputVec(2);
        z(i,1) = outputVec(3);
    
        
    end
    disp(size(x))
    disp(size(y))
    disp(size(z))



end
