% Define the sequence of values
sequence = [0, 1, 2, 3, 4, 5, 6];

% Repeat the sequence to create an array of desired length (e.g., 500)
repeatedSequence = repmat(sequence, 1, ceil(500/length(sequence)));

% Trim the excess elements to match the desired length
repeatedSequence = repeatedSequence(1:500);

% Convert to a categorical array
YTrain = categorical(repeatedSequence);
disp(YTrain)