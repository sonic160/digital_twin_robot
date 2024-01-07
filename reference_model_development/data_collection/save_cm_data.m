function save_cm_data(app)
    % Check if "collected_data" folder exists, if not, create it
    if ~exist('collected_data', 'dir')
        mkdir('collected_data');
    end

    % Create a subfolder with the current time as the folder name
    currentTime = datetime('now');
    currentTime.Format = 'yyyyMMdd_HHmmSS';
    currentTime = string(currentTime);
    subfolderPath = fullfile('collected_data', currentTime);
    if ~exist(subfolderPath, 'dir')
        mkdir(subfolderPath);
    end

    %% This section save the data and export to .csv file.
    for motor_id = 1:6
        file_name = fullfile(subfolderPath, ['data_motor_' num2str(motor_id) '.csv']);
        % Extract a time array from the 'app' object from the 1st element to the 'idx' element.
        time = app.time_array(1:app.idx)';            
        % Extract data related to 'motor_1' from the 'app' object up to the 'idx' element.
        tmp_data = app.data_motor(1:app.idx, :, motor_id);            
        % Extract the 'position,' 'temperature,' and 'voltage' columns from 'tmp_data.'
        position = tmp_data(:, 1);
        temperature = tmp_data(:, 2);
        voltage = tmp_data(:, 3);
        label = app.is_failure(1:app.idx, motor_id);
        % Create a table from the extracted data columns: time, position, temperature, voltage.
        table_motor_1 = table(time, position, temperature, voltage, label);      
        % Write the table to a CSV file named 'data_motor_1.csv'.
        writetable(table_motor_1, file_name);
    end
end