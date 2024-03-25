function connect_and_monitor(app)
    % Shutdown ros nodes, if any.
   rosshutdown;               
   % Connect to the robot: The ip address is 192.168.0.103.
   rosinit('192.168.0.103', 11311)  
   % rosinit('192.168.1.57', 11311)

   % Set this ip address to be the fake robot's, if you are
   % using the offline testing code.
   % rosinit('192.168.1.14', 11311)
   % rosinit('192.168.1.115', 11311)
   % rosinit('138.195.233.201', 11311)

   % Create a subscriber to the 'condition_monitoring' topic
   my_callback = @(src, msg)func_receive_msg(app, src, msg);
   app.sub = rossubscriber('/condition_monitoring', my_callback);
end