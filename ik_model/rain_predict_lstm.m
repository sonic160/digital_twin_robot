% Step 1: Generate the dataset
num_time_series = 4000;
time_series_length = 1000;
num_classes = 4;

% Parameters
cellArray=load('.\cellArray500interpolatesshapes.mat');
sizearray = size(cellArray);
numSeq = sizearray(1);      % Number of sequences
numFeatures = 4;   % Number of features
maxSeqLength = 1000; % Maximum sequence length

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
%disp(categoricalSequence);
totalElements = numel(categoricalSequence);
indexToKeep = round(0.8 * totalElements);

totalCells = numel(cellArray);
index = round(0.8 * totalCells);
XTrain = cellArray(1:index);
XVal = cellArray(index:totalCells);
YTrain = categoricalSequence(1:indexToKeep);
YVal =categoricalSequence(indexToKeep:totalElements);

%numFeatures = size(XTrain{1},1);
%legend("Feature " + string(1:numFeatures),Location="northeastoutside")
% Display the generated data

miniBatchSize = 128;%UTILITE
% Step 2: Define the neural network

inputSize = 6;
numHiddenUnits = 150;
numClasses = 4;

layers =[
    sequenceInputLayer(inputSize) 
    % Bidirectional LSTM layers
    bilstmLayer(numHiddenUnits, 'OutputMode', 'sequence')
    bilstmLayer(numHiddenUnits, 'OutputMode', 'sequence')
    fullyConnectedLayer(numHiddenUnits,'last')
    dropoutLayer(0.2)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
];
options = trainingOptions("sgdm", ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropFactor=0.2, ...
    LearnRateDropPeriod=5, ...
    MaxEpochs=200, ...
    MiniBatchSize=128, ...
    ValidationData={XVal,YVal}, ... %new
    ValidationFrequency=20, ...     %new
    SequenceLength="longest", ...
    L2Regularization = 0.0001, ...  %new
    Shuffle="once", ...
    Verbose=0, ...
    Plots="training-progress");
% options = trainingOptions("adam", ...
%     ExecutionEnvironment="cpu", ...
%     GradientThreshold=1, ...
%     MaxEpochs=200, ...
%     MiniBatchSize=miniBatchSize, ...
%     ValidationData={XVal,YVal}, ... %new
%     ValidationFrequency=20, ...     %new
%     SequenceLength="longest", ...
%     L2Regularization = 0.0001, ...  %new
%     Shuffle="once", ...
%     Verbose=0, ...
%     Plots="training-progress");


net3 = trainNetwork(XTrain,YTrain,layers,options);
save('lstmv3_2bilayers_500interpolate_200hiddenunnit_dropout0_2_alr_128batch.mat','net3')
% Make predictions on the validation set
YPred = predict(net3, XVal);

% Find the column index of the maximum probability for each row
[~, predictedClass] = max(YPred, [], 2);

% Create a categorical array from the predicted class indices
categoryNames = cellstr(num2str((0:max(predictedClass))'));  % Assuming classes are 0-based
categoricalPred = categorical(predictedClass - 1, 0:max(predictedClass), categoryNames);

% Compute confusion matrix
C = confusionmat(YVal, categoricalPred);

% Display confusion chart
figure
confusionchart(YVal, categoricalPred,'RowSummary','row-normalized');

precision = diag(C) ./ sum(C, 1)';
recall = diag(C) ./ sum(C, 2);
f1Score = 2 * (precision .* recall) ./ (precision + recall);

% Display the results
disp('Class   Precision   Recall   F1 Score');
disp([transpose(1:size(C, 1)), precision, recall, f1Score]);

for i = 0:numClasses-1
    % Convert true labels to binary
    YTruBinary = ismember(YVal, num2str(i));
    
    % Extract predicted scores for the current class
    %YPredBinary =ismember(categoricalPred, num2str(i));
    YpredROC=YPred(:,i+1);
    
    % Compute ROC curve
    [X, Y, ~, AUC] = perfcurve(YTruBinary, YpredROC, 1);
    
    % Plot ROC curve for the current class
    plot(X, Y, 'DisplayName', ['Class ' num2str(i) ' (AUC = ' num2str(AUC) ')']);
    
    hold on;
end

