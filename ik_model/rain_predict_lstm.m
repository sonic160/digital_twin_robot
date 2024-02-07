% Step 1num_time_series: Generate the dataset
num_classes = 13;
numClasses = num_classes;

% Parameters
struc=load('./cellArray402_circle_line_interpolation_motor123error0001_moy_600_0203.mat');
%cArray=struc.cellArray;
cArray=struc.cellArray;
% cArray=struc;
sizearray = size(cArray);
numSeq = sizearray(1); % Number of sequences
disp(numSeq)
%length of the testingdata
test_len=1000;

% Size treatment
numberofcells=numel(cArray);
maxsize = size(cArray{1}, 2);
multfactor = maxsize / test_len;

% Create a new cell array to store modified data
 modifiedCellArray = cArray;

% Process each cell in the original array
if multfactor ~= 1
    modifiedCellArray = cell(1, multfactor * maxsize);
    for k = 1:numberofcells
        % Get the data from the original cell
        originalCell = cArray{k};
        for i = 1:multfactor
            acell = originalCell(:, (i - 1) * test_len + 1 : i * test_len);
            % Assign the cells to the modified cell array
            modifiedCellArray{(k - 1) * multfactor + i} = acell;
        end
    end
end

% %size treatment
% maxsize=numel(cArray);
% modifiedCellArray = cell(size(cArray));
% temp_len=100;
% for k = 1:maxsize
%     % Get the data from the original cell
%     originalCell = cArray{k};
% 
%     % Truncate the data to have dimensions of 6-by-temp_len
%     truncatedData = originalCell(:, 1:temp_len);
% 
%     % Assign the truncated data to the corresponding cell in the new array
%     modifiedCellArray{k} = truncatedData;
% end


% % Create a new cell array to store modified data
% modifiedCellArray = cell(1, maxsise/temp_len * maxsize);
% 
% % Process each cell in the original array
% for k = 1:maxsize
%     % Get the data from the original cell
%     originalCell = cArray{k};
% 
%     % Create the first cell with the first temp_len columns
%     firstCell = originalCell(:, 1:temp_len);
% 
%     % Create the second cell with the remaining columns
%     secondCell = originalCell(:, temp_len+1:end);
% 
%     % Assign the cells to the modified cell array
%     modifiedCellArray{2*k-1} = firstCell;
%     modifiedCellArray{2*k} = secondCell;
% end



% Generate random sequences
%XTrain = cell(numSeq, 1);
%YTrain = categorical(randi(6, numSeq, 1)); % Random labels from 1 to 6

% %{for i = 1:numSeq
%     seqLength = 1000;  Random sequence length between 5 and maxSeqLength
%    XTrain{i} = randn(numFeatures, seqLength); % Random feature values
%     Ytr
%     end
%     %}





% Generate the pattern
pattern = mod(0:numSeq-1, numClasses);
% Create the categorical sequence
categoricalSequence = categorical(pattern, 0:numClasses-1);
% Repeat each category in categoricalSequence by multfactor times
repeatedSequence = repelem(categoricalSequence, multfactor);

totalElements = numel(repeatedSequence);
indexToKeep = round(0.8 * totalElements);

totalCells = numel(modifiedCellArray);
index = round(0.8 * totalCells);

XTrain = modifiedCellArray(1:index);
XVal = modifiedCellArray(index+1:totalCells);
YTrain = repeatedSequence(1:indexToKeep);
YVal =repeatedSequence(indexToKeep+1:totalElements);


miniBatchSize = 64;
% Step 2: Define the neural network

inputSize = 6;
numHiddenUnits = 150;



layers = [
    sequenceInputLayer(inputSize, 'Name', 'inputFEN')
    bilstmLayer(numHiddenUnits, 'OutputMode', 'sequence')
    bilstmLayer(numHiddenUnits, 'OutputMode', 'sequence')
    convolution1dLayer(2,5,'Stride',2,'Padding',1)
    maxPooling1dLayer(2,'Stride',3,'Padding',1)
    convolution1dLayer(5, 32, 'Padding', 'same', 'Stride', 2)
    globalAveragePooling1dLayer('Name', 'GlobalAveragePoolingfcn')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
];

options = trainingOptions("adam", ...
    ExecutionEnvironment="gpu", ...
    GradientThreshold=1, ...
    MaxEpochs=400, ...
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



save('gated_transformer_600_c_l_i_motorerror_00010203_0123_1000.mat','net')
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

