% This script configurate an enviorenment that generates a pseudo robot so
% that you can take your program offline, without accessing to the real
% robot.

clear; clc; rosshutdown;

% Initialize ROS node
rosinit('localhost'); % Use the appropriate ROS master URI if needed

% Call the condition_monitoring function
path = '../data_collection/collected_data/task_fault/';
filenames = {'data_motor_1.csv', 'data_motor_2.csv', 'data_motor_3.csv', 'data_motor_4.csv', 'data_motor_5.csv', 'data_motor_6.csv'};

condition_monitoring(path, filenames);


function condition_monitoring(path, filenames)
% This function reads the data in [path filenames] and sends them through
% the conditino_monitoring topic.
% The data are sent recursively: If a seqeunce is finished, it will restart
% from the beginning.
% When a new sequence starts, a msg of "Prepare to receive the data!" will
% be sent first.

    pub = rospublisher('/condition_monitoring', 'std_msgs/String');
    msg = rosmessage(pub);

    rate = rosrate(10); % 10 Hz

    % Read the true data in the csv files.
    cm_data = cell(1, 6);
    for i = 1:6
        tmp = readtable([path filenames{i}]);
        cm_data{i} = tmp;
    end

    % Prepare position, temperature and voltage.
    n_s = size(cm_data{1}, 1); % Number of data points.
    % Initial values.
    position = zeros(n_s, 6);
    temperature = zeros(n_s, 6);
    voltage = zeros(n_s, 6);
    % Read each table:
    for i = 1:6
        tmp = cm_data{1, i};
        position(:, i) = tmp.position;
        temperature(:, i) = tmp.temperature;
        voltage(:, i) = tmp.voltage;
    end

    % Sending the testing sequence recursively.
    while (1)
    idx = 0;
        while(idx<=n_s)
            if idx == 0 % Send a starting signal.
                % Create the message
                msg.Data = 'Prepare to receive the data!';
            else
                % Get the bus servo status
                pos = position(idx, :);
                temp = temperature(idx, :);
                vol = voltage(idx, :);
        
                % Create the message
                msg.Data = sprintf('position: %s\n temperature: %s\n voltage: %s\n \n', ...
                    mat2str(pos), mat2str(temp), mat2str(vol));
            end

            % Publish the message
            send(pub, msg);
    
            idx = idx + 1;        
            waitfor(rate);
        end
    end
end
