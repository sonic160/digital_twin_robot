function saveAndPlotMotorData(app, motor_id, plt_handles)                   
    % Get tnow.
    tnow = app.idx;
    % Loop for position, temperature and voltage.
    for i = 1:3
        % Read the current measurement.
        y = app.data_motor(app.idx, i, motor_id);
        % Get the handle to the plot.
        plt = plt_handles{i};

        % Get the current x-axis limits of the app's UIAxes_pos.
        xLL = plt.XLim(1);
        xUL = plt.XLim(2);

        % Check if the current time (tnow) is out of bounds.
        if (tnow > xUL)
            % If it's out of bounds, add 15 seconds to the x-axis limits.
            xtra = xUL - xLL;
            plt.XLim = [xLL + xtra, xUL + xtra];
            % Adjust the x-axis ticks to maintain readability.
            plt.XTick = linspace(xLL + xtra, xUL + xtra, 5);
        end

        % Plot the position data at the current time tnow.
        plot(plt, tnow, y, ".b");

        % Adjust the y-axis limits to show variations better.
        if app.idx <= 2
            yLL = max(y*.8, 0);
            yUL = y*1.2;
        else
            y_lim = plt.YLim;
            yLL = y_lim(1);
            yUL = y_lim(2);
            if y < yLL
                yLL = .95*y;
            else 
                if y > yUL
                    yUL = 1.05*y;
                end
            end
        end
        plt.YLim = [yLL, yUL];
        plt.YTick = round(linspace(yLL, yUL, 5), 2, "significant");
    end            
end