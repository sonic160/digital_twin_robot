function func_receive_msg(app, ~, msg)
    % This is the callback function for the subscriber.
    % Will be executed whenever a msg is published.
    % The received msg will be automatically given to this func as the
    % input "msg".

    % If the maximal data storage is reached
    if app.idx == app.data_limit
        % Close the ROS connection
        rosshutdown;
        % Save the collected data.
        save_cm_data(app)                      
    else    
        % Update the current index.
        app.idx = app.idx + 1;
    
        % Get current time
        % Only keep the last 5 digits for the seconds, and transform from
        % nano seconds into seconds.
        tnow = mod(msg.Header.Stamp.Sec, 1e5) + msg.Header.Stamp.Nsec*1e-9; 
        % Save the data collection time.
        app.time_array{app.idx} = tnow;
    
        % Parse the data to get a matrix of monitoring data.
        tmp_data = [msg.Position'; msg.Temperature'; msg.Voltage'];
        % Save the CM data.
        app.data_motor(app.idx, :, :) = tmp_data;
    
        % Update the plots.
        % Hanldes of the Figures.
        plt_handles = {{app.UIAxes_pos_1, app.UIAxes_temp_1, app.UIAxes_vol_1}, ...
            {app.UIAxes_pos_2, app.UIAxes_temp_2, app.UIAxes_vol_2}, ...
            {app.UIAxes_pos_3, app.UIAxes_temp_3, app.UIAxes_vol_3}, ...
            {app.UIAxes_pos_4, app.UIAxes_temp_4, app.UIAxes_vol_4}, ...
            {app.UIAxes_pos_5, app.UIAxes_temp_5, app.UIAxes_vol_5}, ...
            {app.UIAxes_pos_6, app.UIAxes_temp_6, app.UIAxes_vol_6}};
        for i = 1:6
            saveAndPlotMotorData(app, i, plt_handles{i});
        end
    end
end