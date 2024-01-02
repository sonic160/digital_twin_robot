% This script shows how to read the condition-monitoring data from the
% "/condition-monitoring" topic using echo method. Then, we write a parse
% function to get the numerical values from the origianl message.

clear; clc; rosshutdown;

% Get the broadcast data here:
% Make sure the ip address is the one of the robot.
rosinit('192.168.1.14', 11311) 
msg = rostopic("echo", "/condition_monitoring");

% Retrieve info from the message
% Split the string into lines
data_lines = strsplit(msg.Data, '\n');

% Initialize arrays to store the data
position_data = [];
temperature_data = [];
voltage_data = [];

% Iterate through each line and extract the values
for i = 1:length(data_lines)
    line = data_lines{i};
    % Split each line by ':' to separate the key and values
    parts = strsplit(line, ':');
    if length(parts) == 2
        key = strtrim(parts{1});
        values = str2num(strtrim(strrep(strrep(parts{2}, '[', ''), ']',''))); %# Remove '[' and convert to numeric array

        % Check which key corresponds to the data and store it in the respective array
        if strcmp(key, 'position')
            position_data = values;
        elseif strcmp(key, 'temperature')
            temperature_data = values;
        elseif strcmp(key, 'voltage')
            voltage_data = values;
        end
    end
end

data = [position_data; temperature_data; voltage_data]