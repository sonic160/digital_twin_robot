% Set the range
lower_bound = 0;
upper_bound = 499;

% Number of indices to select
num_indices = 167;

% Generate random indices
random_indices1 = randperm(upper_bound - lower_bound + 1, num_indices) + lower_bound - 1;
%disp('random_indices1');
%disp(random_indices1(1:12));
% Create an array
original_array1 = [];

% Modify the array based on the selected indices
for i = 1:length(random_indices1)
    index = 4*random_indices1(i);
    
    % Add i+1, i+2, and i+3 immediately after index i
    original_array1 = [original_array1, index + 1, index + 2, index + 3, index + 4];
end

% Display the modified array
%disp('original_array1');
%disp(original_array1(1:12));
% Set the range
lower_bound = 0;
upper_bound = 999;

% Number of indices to select
num_indices = 333;

% Generate random indices
random_indices2 = randperm(upper_bound - lower_bound + 1, num_indices) + lower_bound - 1;
%disp('random_indices2');
%disp(random_indices2(1:12));

% Create an array
original_array2 = [];

% Modify the array based on the selected indices
for i = 1:length(random_indices2)
    index = 4*random_indices2(i);
    
    % Add i+1, i+2, and i+3 immediately after index i
    original_array2 = [original_array2, index + 1, index + 2, index + 3, index + 4];
end

% Display the modified array
% disp('original_array2');
% disp(original_array2(1:12));

% Select elements from CA
selectedElementsCA = CA.cellArray(original_array1);

% Select elements from CB
selectedElementsCB = CB.cellArray(original_array2);

% Concatenate the selected elements
concatenatedSelectedElements = [selectedElementsCA; selectedElementsCB];

% Create CD structure
CD = struct('cellArray', {concatenatedSelectedElements});

% Display the size of CD to verify
disp(size(CD.cellArray));