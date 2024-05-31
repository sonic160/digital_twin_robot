function save_cm_data(app)
    % Check if "collected_data" folder exists, if not, create it
    folder_name = 'collected_data_students'
    if ~exist(folder_name, 'dir')
        mkdir(folder_name);
    end

    % Create a subfolder with the current time as the folder name
    currentTime = datetime('now');
    currentTime.Format = 'yyyyMMdd_HHmmSS';
    currentTime = string(currentTime);
    subfolderPath = fullfile(folder_name, currentTime);
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
        % Inject failure
        temperature = inject_failure(temperature, label);
        % Create a table from the extracted data columns: time, position, temperature, voltage.
        table_motor_1 = table(time, position, temperature, voltage, label);      
        % Write the table to a CSV file named 'data_motor_1.csv'.
        writetable(table_motor_1, file_name);
    end
end


function temperature = inject_failure(temperature, label)
    % Get the sequence where failure is injected.
    tmp_temp = temperature(label==1);
    % Length of the sequence.
    n_seq = length(tmp_temp);
    % Decide the sequence where temperature rise: We assume temperature
    % rise is 2 times faster as temperature drop.
    n_rise = floor(n_seq/3);
    if ~isempty(tmp_temp) 
        % Get the starting and ending temperature
        temp_start = tmp_temp(1);
        temp_end = tmp_temp(end);

        % Generate a random highest temperature
        temp_high = max(temp_start, temp_end) + randi([2, 10]);
        % Generate the temperature rise
        step_size_rise = floor(n_rise/(temp_high-temp_start+1));
        for i = 1:temp_high-temp_start
            tmp_temp((i-1)*step_size_rise+1:i*step_size_rise) = temp_start+i-1;
        end
        tmp_temp(i*step_size_rise+1:n_rise) = temp_high;

        % Generate temperature decrease
        step_size_down = floor(2*n_rise/(temp_high-temp_end));
        for i = 1:temp_high-temp_end-1
            tmp_temp(n_rise+1+(i-1)*step_size_down:n_rise+i*step_size_down) = temp_high-i;
        end
        tmp_temp(n_rise+i*step_size_down+1:end) = temp_end;       
    end
    temperature(label==1) = tmp_temp;
end