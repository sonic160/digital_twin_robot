% Step 1: Generate the dataset
num_time_series = 6;
time_series_length = 1000;
num_classes = 4;

% Parameters
numSeq = 20;      % Number of sequences
numFeatures = 4;   % Number of features
maxSeqLength = 1000; % Maximum sequence length

% Generate random sequences
%XTrain = cell(numSeq, 1);
YTrain = categorical(randi(6, numSeq, 1)); % Random labels from 1 to 6

% %{for i = 1:numSeq
%     seqLength = 1000;  Random sequence length between 5 and maxSeqLength
%    XTrain{i} = randn(numFeatures, seqLength); % Random feature values
%     Ytr
%     end
%     %}

numClasses = 4;  % Number of classes

% Generate categorical sequence
pattern = mod(0:numSeq-1, numClasses);
categoricalSequence = categorical(pattern, 0:numClasses-1);
disp(categoricalSequence);
totalElements = numel(categoricalSequence);
indexToKeep = round(0.8 * totalElements);

totalCells = numel(cellArray);
index = round(0.8 * totalCells);
XTrain = cellArray(1:index);
XVal = cellArray(index:totalCells);
YTrain = categoricalSequence(1:indexToKeep);
YVal =categoricalSequence(indexToKeep:totalElements);
figure
plot(XTrain{1}')
xlabel("Time Step")
title("Training Observation 1")
%numFeatures = size(XTrain{1},1);
%legend("Feature " + string(1:numFeatures),Location="northeastoutside")
% Display the generated data

miniBatchSize = 128;
% Step 2: Define the neural network

inputSize = 6;
numHiddenUnits = 100;
numClasses = 4;

layers = [
    sequenceInputLayer(inputSize)
    bilstmLayer(numHiddenUnits, 'OutputMode', 'last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
];

options = trainingOptions("adam", ...
    ExecutionEnvironment="cpu", ...
    GradientThreshold=1, ...
    MaxEpochs=50, ...
    MiniBatchSize=miniBatchSize, ...
    ValidationData={XVal,YVal}, ... %new
    ValidationFrequency=20, ...     %new
    SequenceLength="longest", ...
    L2Regularization = 0.0001, ...  %new
    Shuffle="once", ...
    Verbose=0, ...
    Plots="training-progress");


net = trainNetwork(XTrain,YTrain,layers,options);



