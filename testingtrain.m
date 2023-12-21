% Step 1: Generate the dataset
num_time_series = 6;
time_series_length = 1000;
num_classes = 4;

% Parameters
numSeq = 2000;      % Number of sequences
numFeatures = 2;   % Number of features


% Generate random sequences
%XTrain = cell(numSeq, 1);
%YTrain = categorical(randi(6, numSeq, 1)); % Random labels from 1 to 6

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

%%%%%%%%%%%%%%%%%

% % Assuming you have XTrain, YTrain, XVal, YVal
% 
% % Merge classes 1, 2, 3 into a single class 1
% YTrainMerged = YTrain;
% YTrainMerged(YTrainMerged == '1' | YTrainMerged == '2' | YTrainMerged == '3') = '1';
% 
% YValMerged = YVal;
% YValMerged(YValMerged == '1' | YValMerged == '2' | YValMerged == '3') = '1';
% 
% % Rebalance the dataset
% class1Indices = find(YTrainMerged == '1');
% class0Indices = find(YTrainMerged == '0');
% 
% % Randomly undersample class 1 to balance with class 0
% undersampledClass1Indices = datasample(class1Indices, numel(class0Indices), 'Replace', false);
% 
% % Combine undersampled class 1 with all instances of class 0
% undersampledIndices = [undersampledClass1Indices; class0Indices];
% 
% % Shuffle the indices to maintain randomness
% undersampledIndices = undersampledIndices(randperm(length(undersampledIndices)));
% 
% % Apply the undersampled indices to XTrain and YTrain
% XTrainUndersampled = XTrain(undersampledIndices);
% YTrainUndersampled = YTrainMerged(undersampledIndices);
% 
% % Display the class distribution after rebalancing
% disp('Class distribution after rebalancing:');
% disp(['Class 0 count: ' num2str(sum(YTrainUndersampled == '0'))]);
% disp(['Class 1 count: ' num2str(sum(YTrainUndersampled == '1'))]);

% Now, you can use XTrainUndersampled and YTrainUndersampled for training.
% You may also apply the same undersampling strategy to the validation set if needed.





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
    
    % Bidirectional LSTM layers
    bilstmLayer(numHiddenUnits, 'OutputMode', 'sequence')
    lstmLayer(numHiddenUnits, 'OutputMode', 'sequence')
    fullyConnectedLayer(numHiddenUnits)
    dropoutLayer(0.3)
    lstmLayer(numHiddenUnits, 'OutputMode', 'last')
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