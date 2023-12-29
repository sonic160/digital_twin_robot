% Step 1num_time_series: Generate the dataset
num_classes = 4;
numClasses = num_classes;

% Parameters
struc=load('cellArray500_circle_line_interpolates_shapes_motor123error00_reducedsize6_500.mat');
%cArray=struc.cellArray;
cArray=struc.newCell;
% cArray=struc;
sizearray = size(cArray);

numSeq = sizearray(1);      % Number of sequences


% Generate random sequences
%XTrain = cell(numSeq, 1);
%YTrain = categorical(randi(6, numSeq, 1)); % Random labels from 1 to 6

% %{for i = 1:numSeq
%     seqLength = 1000;  Random sequence length between 5 and maxSeqLength
%    XTrain{i} = randn(numFeatures, seqLength); % Random feature values
%     Ytr
%     end
%     %}

pattern = mod(0:numSeq-1, numClasses);
categoricalSequence = categorical(pattern, 0:numClasses-1);
totalElements = numel(categoricalSequence);
indexToKeep = round(0.8 * totalElements);

totalCells = numel(cellArray);
index = round(0.8 * totalCells);
XTrain = cellArray(1:index);
XVal = cellArray(index:totalCells);
YTrain = categoricalSequence(1:indexToKeep);
YVal =categoricalSequence(indexToKeep:totalElements);


miniBatchSize = 64;
% Step 2: Define the neural network

inputSize = 6;
numHiddenUnits = 150;



layers = [
    sequenceInputLayer(inputSize)
    
    % Bidirectional LSTM layers
    bilstmLayer(numHiddenUnits, 'OutputMode', 'sequence')
    bilstmLayer(numHiddenUnits, 'OutputMode', 'sequence')
    fullyConnectedLayer(numHiddenUnits)
    dropoutLayer(0.2)
    lstmLayer(numHiddenUnits, 'OutputMode', 'last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
];

options = trainingOptions("adam", ...
    ExecutionEnvironment="gpu", ...
    GradientThreshold=1, ...
    MaxEpochs=100, ...
    MiniBatchSize=miniBatchSize, ...
    ValidationData={XVal,YVal}, ... %new
    ValidationFrequency=20, ...     %new
    SequenceLength="longest", ...
    L2Regularization = 0.0001, ...  %new
    Shuffle="once", ...
    Verbose=0, ...
    Plots="training-progress");

% options = trainingOptions("sgdm", ...
%     ExecutionEnvironment="cpu", ...  % Specify CPU execution
%     LearnRateSchedule="piecewise", ...
%     LearnRateDropFactor=0.2, ...
%     LearnRateDropPeriod=5, ...
%     MaxEpochs=200, ...
%     MiniBatchSize=128, ...
%     ValidationData={XVal,YVal}, ...
%     ValidationFrequency=20, ...
%     SequenceLength="longest", ...
%     L2Regularization = 0.0001, ...
%     Shuffle="once", ...
%     Verbose=0, ...
%     Plots="training-progress");


net = trainNetwork(XTrain,YTrain,layers,options);



save('lstmv3_2bilayers_150_line_circle_interpolate_motorerror00_0123_reduced6_500_150hiddenunnit_dropout0_2_alr_128batch.mat','net')
% Make predictions on the validation set
YPred = predict(net, XVal);

% Find the column index of the maximum probability for each row
[~, predictedClass] = max(YPred, [], 2);

% Create a categorical array from the predicted class indices
categoryNames = cellstr(num2str((0:max(predictedClass))'));  % Assuming classes are 0-based
categoricalPred = categorical(predictedClass - 1, 0:max(predictedClass), categoryNames);

% Compute confusion matrix
C = confusionmat(YVal, categoricalPred)

% Display confusion chart
figure
confusionchart(YVal, categoricalPred,'RowSummary','row-normalized');
title('Confusion Matrix');


precision = diag(C) ./ sum(C, 1)';
recall = diag(C) ./ sum(C, 2);
f1Score = 2 * (precision .* recall) ./ (precision + recall);


% Display the results
disp('Class   Precision   Recall   F1 Score');
disp([transpose(1:size(C, 1)), precision, recall, f1Score]);

% Create a new figure for ROC curves
figure

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

% Add labels and legend
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curves for Multi-Class Classification');
legend('show');

% Create a new figure for Precision-Recall curves
figure

for i = 0:numClasses-1
    % Convert true labels to binary
    YTruBinary = ismember(YVal, num2str(i));
    
    % Extract predicted scores for the current class
    YpredROC = YPred(:, i+1);
    
    % Compute Precision-Recall curve
    [precision, recall, ~, AUC] = perfcurve(YTruBinary, YpredROC, 1, 'xCrit', 'reca', 'yCrit', 'prec');
    
    % Plot Precision-Recall curve for the current class
    plot(recall, precision, 'DisplayName', ['Class ' num2str(i) ' (AUC = ' num2str(AUC) ')']);
    
    hold on;
end
% Add labels and legend
xlabel('Recall');
ylabel('Precision');
title('Precision-Recall Curves for Multi-Class Classification');
legend('Location', 'Best');
hold off; % Stop holding onto the current plot

