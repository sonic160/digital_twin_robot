% This script configurate an enviorenment that generates a pseudo robot so
% that you can take your program offline, without accessing to the real
% robot.

clear; clc; rosshutdown;

% Initialize ROS node
rosinit('localhost'); % Use the appropriate ROS master URI if needed

% Call the condition_monitoring function
condition_monitoring();


function condition_monitoring()
    pub = rospublisher('/condition_monitoring', 'std_msgs/String');
    msg = rosmessage(pub);

    rate = rosrate(10); % 10 Hz

    while(1)
        % Get the bus servo status
        [pos, temp, voltage] = getBusServoStatus();

        % Create the message
        msg.Data = sprintf('position: %s\n temperature: %s\n voltage: %s\n \n', ...
            mat2str(pos), mat2str(temp), mat2str(voltage));

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

