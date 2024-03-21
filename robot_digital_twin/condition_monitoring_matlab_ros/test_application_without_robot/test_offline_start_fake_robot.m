% This script configurate an enviorenment that generates a pseudo robot so
% that you can take your program offline, without accessing to the real
% robot.

clear; clc; rosshutdown;

% Initialize ROS node
rosinit('localhost'); % Use the appropriate ROS master URI if needed

% Call the condition_monitoring function
condition_monitoring();


function condition_monitoring()
    pub = rospublisher('/condition_monitoring', 'cm/msg_cm');
    
    msg = rosmessage(pub);
    msg.Name = {'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'r_joint'};
    msg.Header.FrameId = 'not_relervant';
    msg.Header.Stamp.Sec = 1e5;
    msg.Header.Stamp.Nsec = 1e3;
    
    rate = rosrate(10); % 10 Hz

    while(1)
        % Get the bus servo status
        [pos, temp, voltage] = getBusServoStatus();

        % Create the message
        msg.Position = pos;
        msg.Temperature = temp;
        msg.Voltage = voltage;

        % Publish the message
        send(pub, msg);
        
        waitfor(rate);
    end
end


function [pos, temp, voltage] = getBusServoStatus()
    % Initialize arrays
    pos = zeros(1, 6);
    temp = zeros(1, 6);
    voltage = zeros(1, 6);

    % Loop to read the status of the six motors
    for motor_idx = 1:6
        % Replace the following lines with your actual implementation
        pos(motor_idx) = 500 + 10*rand();  % Replace with Board.getBusServoPulse(motor_idx);
        temp(motor_idx) = 30 + 5*rand(); % Replace with Board.getBusServoTemp(motor_idx);
        voltage(motor_idx) = 7000 + 10*rand(); % Replace with Board.getBusServoVin(motor_idx);
    end
end

