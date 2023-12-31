% Step 1num_time_series: Generate the dataset
num_classes = 4;
numClasses = num_classes;

% Parameters
struc=load('../cellArray1000.mat');
%cArray=struc.cellArray;
cArray=struc.cellArray;
% cArray=struc;
sizearray = size(cArray);
numSeq = sizearray(1); % Number of sequences
disp(numSeq)


% Initialize variables to store results
%valset
bestModelF1 = 0;
bestModelIdx = 0;
allF1Scores = zeros(1, 10);
bestModelF1perclass =zeros(1, numClasses);
bestModelIdxperclass =zeros(1, numClasses);
allF1Scoresperclass = zeros(10, numClasses);

%dataset1

% modelindex * multfactor
modelsdataset1meanF1matrix=zeros(10,10);
% modelindex * [(c0,c1,c2,c3)*multfactor]
allF1scoresdataset1=zeros(10,40);

%dataset2
modelsdataset2meanF1matrix = zeros(10, 10);
allF1scoresdataset2 = zeros(10, 40);

%dataset3
modelsdataset3meanF1matrix = zeros(10, 10);
allF1scoresdataset3 = zeros(10, 40);

% Specify the directory for saving gathered data
gatheredDataDir = 'GatheredData/training_1000_line_circles';

% Create directories if they don't exist
if ~exist(gatheredDataDir, 'dir')
    mkdir(gatheredDataDir);
end
%should be named time factor
for index0=1:10
    disp("---------------------------------------------------------")
    disp("---------------------------------------------------------")
    disp("---------------------------------------------------------")
    disp(['We are working with model' num2str(index0) ':']);
    disp("---------------------------------------------------------")
    disp("---------------------------------------------------------")
    disp("---------------------------------------------------------")
    %length of the testingdata
    test_len=index0*100;
    multfactor=floor(maxsize/test_len);
    
    % Size treatment
    numberofcells=numel(cArray);
    maxsize = size(cArray{1}, 2);

   % Create a new cell array to store modified data
   modifiedCellArray = cell(1, numberofcells * multfactor);
    
   % Process each cell in the original array
   for k = 1:numberofcells
        % Get the data from the original cell
        originalCell = cArray{k};
        
        % Repeat and slice the data to fit the desired length
        for i = 1:multfactor
            startIdx = (i - 1) * test_len + 1;
            endIdx = min(i * test_len, maxsize);
            acell = originalCell(:, startIdx:endIdx);
            
            % Assign the cells to the modified cell array
            modifiedCellArray{(k - 1) * multfactor + i} = acell;
        end
    end
    
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
   
    %option initalisation

    options = trainingOptions("adam", ...
    ExecutionEnvironment="gpu", ...
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
     %Directory where the saves will happen 
    
    
    save(sprintf('lstmv3_2bilayers_1000_line_circle_motorerror00_0123_reduced_%d_6_%d_150hiddenunnit_dropout0_2_alr_128batch.mat', index0, test_len), 'net');
    

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
    disp(['Results for Model ' num2str(index0) ':']);
    disp('Class   Precision   Recall   F1 Score');
    disp([transpose(1:size(C, 1)), precision, recall, f1Score]);
    
    % Save confusion matrix for the current model
    modelDir = fullfile(gatheredDataDir, sprintf('model_%d', index0));
    confusionMatrixDir = fullfile(modelDir, 'validation_data', 'confusion_matrix');
    if ~exist(confusionMatrixDir, 'dir')
        mkdir(confusionMatrixDir);
    end
    save(fullfile(confusionMatrixDir, 'confusion_matrix.mat'), 'C');


    f1Score=f1Score(1:4,:);
    f1Score=f1Score';
    % Store F1 score for the current model
    allF1Scores(index0) = mean(f1Score);
    % Store per class F1 score for the current model
    allF1Scoresperclass(index0, :) = f1Score;

    % Check if current model has the best F1 score
    if mean(f1Score) > bestModelF1
        bestModelF1 = mean(f1Score);
        bestModelIdx = index0;
    end
    % Check if current model has the best per class F1 score for each class
    for classIdx = 1:numClasses
        if f1Score(classIdx) > bestModelF1perclass(classIdx)
            bestModelF1perclass(classIdx) = f1Score(classIdx);
            bestModelIdxperclass(classIdx) = index0;
        end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%
    %DATASET1
    %%%%%%%%%%%%%%%%%%%%%%%%
    

    loadedcell=load('../cellArray500interpolatesshapes.mat');
    cellArray=loadedcell.cellArray;
    %cellArray=loadedcell.CD.cellArray;
    sizearray = size(cellArray);
    numSeq = sizearray(1); % Number of sequences
    disp(numSeq)
    
  
    % Size treatment
    numberofcells=numel(cellArray);
    maxsize = size(cellArray{1}, 2);
    
    
    
    
    for index1= 1:10
        disp("---------------------------------------------------------")
        disp("---------------------------------------------------------")
        disp("---------------------------------------------------------")
        disp(['We are working with model ' num2str(index0) ' on dataset1 version: ' num2str(index1)]);
        disp("---------------------------------------------------------")
        disp("---------------------------------------------------------")
        disp("---------------------------------------------------------")
       test_len1=index1*100;
       multfactor1=floor(maxsize/test_len1);
       % Create a new cell array to store modified data
       modifiedCellArray = cell(1, numberofcells * multfactor1);
        
       % Process each cell in the original array
       for k = 1:numberofcells
            % Get the data from the original cell
            originalCell = cellArray{k};
            
            % Repeat and slice the data to fit the desired length
            for i = 1:multfactor1
                startIdx = (i - 1) * test_len1 + 1;
                endIdx = min(i * test_len1, maxsize);
                acell = originalCell(:, startIdx:endIdx);
                
                % Assign the cells to the modified cell array
                modifiedCellArray{(k - 1) * multfactor1 + i} = acell;
            end
        end
        
        
        numClasses = 4;  % Number of classes
        
        % Generate the pattern
        pattern = mod(0:numSeq-1, numClasses);
        
        % Create the categorical sequence
        categoricalSequence = categorical(pattern, 0:numClasses-1);
        
        % Repeat each category in categoricalSequence by multfactor1 times
        repeatedSequence = repelem(categoricalSequence, multfactor1);
        
        
        XVal = modifiedCellArray;
        YVal = repeatedSequence;
        
        YPred = predict(net, XVal);
        
        % Find the column index of the maximum probability for each row
        [~, predictedClass] = max(YPred, [], 2);
        
        % Create a categorical array from the predicted class indices
        categoryNames = cellstr(num2str((0:max(predictedClass))'));  % Assuming classes are 0-based
        categoricalPred = categorical(predictedClass - 1, 0:max(predictedClass), categoryNames);
        % Compute confusion matrix
        size(YVal)
        size(categoricalPred)
        C = confusionmat(YVal, categoricalPred)
        
        % Display confusion chart
        figure
        confusionchart(YVal, categoricalPred,'RowSummary','row-normalized');
        title('Confusion Matrix');
        
        
        precision = diag(C) ./ sum(C, 1)';
        recall = diag(C) ./ sum(C, 2);
        f1Score = 2 * (precision .* recall) ./ (precision + recall);
        
        
        % Display the results
        disp(['Results for Model ' num2str(index0) ' on dataset1 version ' num2str(index1)]);
        disp('Class   Precision   Recall   F1 Score');
        disp([transpose(1:size(C, 1)), precision, recall, f1Score]);
        
        % Save confusion matrix for the current model
        modelDir = fullfile(gatheredDataDir, sprintf('model_%d', index0));
        confusionMatrixDir = fullfile(modelDir, sprintf('dataset1_%d', index1), 'confusion_matrix');
        if ~exist(confusionMatrixDir, 'dir')
            mkdir(confusionMatrixDir);
        end
        save(fullfile(confusionMatrixDir, 'confusion_matrix.mat'), 'C');
    
    
        f1Score=f1Score(1:4,:);
        f1Score=f1Score';
        % Store mean F1 score for the current model
        modelsdataset1meanF1matrix(index0,index1) = mean(f1Score);
        % Store per class F1 score for the current model
        allF1scoresdataset1(index0, index1*4-3:index1*4) = f1Score;
    end

    %%%%%%%%%%%%%%%%%%%%%%%%
    %DATASET2
    %%%%%%%%%%%%%%%%%%%%%%%%

    loadedcell=load('../cellArray2000_circle_line_interpolatesshapes.mat');
    %cellArray=loadedcell.cellArray;
    cellArray=loadedcell.CD.cellArray;
    sizearray = size(cellArray);
    numSeq = sizearray(1); % Number of sequences
    disp(numSeq)
    
  
    % Size treatment
    numberofcells=numel(cellArray);
    maxsize = size(cellArray{1}, 2);

    for index2= 1:10
        disp("---------------------------------------------------------")
        disp("---------------------------------------------------------")
        disp("---------------------------------------------------------")
        disp(['We are working with model ' num2str(index0) ' on dataset2 version: ' num2str(index2)]);
        disp("---------------------------------------------------------")
        disp("---------------------------------------------------------")
        disp("---------------------------------------------------------")
        test_len2=index2*100;
        multfactor2=floor(maxsize/test_len2);
        % Create a new cell array to r2store modified data
        modifiedCellArray = cell(1, numberofcells * multfactor2);
            % 
        % Process each cell in the original array
        for k = 1:numberofcells
            % Get the data from the original cell
            originalCell = cellArray{k};
            
            % Repeat and slice the data to fit the desired length
            for i = 1:multfactor2
                startIdx = (i - 1) * test_len2 + 1;
                endIdx = min(i * test_len2, maxsize);
                acell = originalCell(:, startIdx:endIdx);
                
                % Assign the cells to the modified cell array
                modifiedCellArray{(k - 1) * multfactor2 + i} = acell;
            end
        end
            
            
        numClasses = 4;  % Number of classes
        
        % Generate the pattern
        pattern = mod(0:numSeq-1, numClasses);
        
        % Create the categorical sequence
        categoricalSequence = categorical(pattern, 0:numClasses-1);
        
        % Repeat each category in categoricalSequence by multfactor2 times
        repeatedSequence = repelem(categoricalSequence, multfactor2);
        
        % Display the result
        %disp(repeatedSequence);
        % 
        % % Generate categorical sequence
        % 
        % pattern = mod(0:numSeq-1, numClasses);
        % categoricalSequence = categorical(pattern, 0:numClasses-1);
        %disp(categoricalSequence);
        
        
        XVal = modifiedCellArray;
        YVal = repeatedSequence;
        
        YPred = predict(net, XVal);
        
        % Find the column index of the maximum probability for each row
        [~, predictedClass] = max(YPred, [], 2);
        
        % Create a categorical array from the predicted class indices
        categoryNames = cellstr(num2str((0:max(predictedClass))'));  % Assuming classes are 0-based
        categoricalPred = categorical(predictedClass - 1, 0:max(predictedClass), categoryNames);
        % Compute confusion matrix
        size(YVal)
        size(categoricalPred)
        C = confusionmat(YVal, categoricalPred)
        
        % Display confusion chart
        figure
        confusionchart(YVal, categoricalPred,'RowSummary','row-normalized');
        title('Confusion Matrix');
        
        
        precision = diag(C) ./ sum(C, 1)';
        recall = diag(C) ./ sum(C, 2);
        f1Score = 2 * (precision .* recall) ./ (precision + recall);
        
        
        % Display the results
        disp(['Results for Model ' num2str(index0) ' on dataset2 version ' num2str(index2)]);
        disp('Class   Precision   Recall   F1 Score');
        disp([transpose(1:size(C, 1)), precision, recall, f1Score]);
        
        % Save confusion matrix for the current model
        modelDir = fullfile(gatheredDataDir, sprintf('model_%d', index0));
        confusionMatrixDir = fullfile(modelDir, sprintf('dataset2_%d', index2), 'confusion_matrix');
        if ~exist(confusionMatrixDir, 'dir')
            mkdir(confusionMatrixDir);
        end
        save(fullfile(confusionMatrixDir, 'confusion_matrix.mat'), 'C');
        f1Score = f1Score(1:4, :);
        f1Score = f1Score';
        modelsdataset2meanF1matrix(index0, index2) = mean(f1Score);
        allF1scoresdataset2(index0, index2 * 4 - 3:index2 * 4) = f1Score;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%
    %DATASET3
    %%%%%%%%%%%%%%%%%%%%%%%%
    disp("---------------------------------------------------------")
    disp("---------------------------------------------------------")
    disp("---------------------------------------------------------")
    disp(['We are working with model ' num2str(index0) ' on dataset3']);
    disp("---------------------------------------------------------")
    disp("---------------------------------------------------------")
    disp("---------------------------------------------------------")

    loadedcell=load('../cellArray1000.mat');
    cellArray=loadedcell.cellArray; 
    %cellArray=loadedcell.CD.cellArray;
    sizearray = size(cellArray);
    numSeq = sizearray(1); % Number of sequences
    disp(numSeq)
    
  
    % Size treatment
    numberofcells=numel(cellArray);
    maxsize = size(cellArray{1}, 2);
       
    for index3= 1:10
        
        test_len3=index3*100;
        multfactor3=floor(maxsize/test_len3);
        % Create a new cell array to r2store modified data
        modifiedCellArray = cell(1, numberofcells * multfactor3);
            
        % Process each cell in the original array
        for k = 1:numberofcells
            % Get the data from the original cell
            originalCell = cellArray{k};
            
            % Repeat and slice the data to fit the desired length
            for i = 1:multfactor3
                startIdx = (i - 1) * test_len3 + 1;
                endIdx = min(i * test_len3, maxsize);
                acell = originalCell(:, startIdx:endIdx);
                
                % Assign the cells to the modified cell array
                modifiedCellArray{(k - 1) * multfactor3 + i} = acell;
            end
        end
            
            
        numClasses = 4;  % Number of classes
        
        % Generate the pattern
        pattern = mod(0:numSeq-1, numClasses);
        
        % Create the categorical sequence
        categoricalSequence = categorical(pattern, 0:numClasses-1);
        
        % Repeat each category in categoricalSequence by multfactor3 times
        repeatedSequence = repelem(categoricalSequence, multfactor3);
        
        % Display the result
        %disp(repeatedSequence);
        % 
        % % Generate categorical sequence
        % 
        % pattern = mod(0:numSeq-1, numClasses);
        % categoricalSequence = categorical(pattern, 0:numClasses-1);
        %disp(categoricalSequence);
        
        
        XVal = modifiedCellArray;
        YVal = repeatedSequence;
        
        YPred = predict(net, XVal);
        
        % Find the column index of the maximum probability for each row
        [~, predictedClass] = max(YPred, [], 2);
        
        % Create a categorical array from the predicted class indices
        categoryNames = cellstr(num2str((0:max(predictedClass))'));  % Assuming classes are 0-based
        categoricalPred = categorical(predictedClass - 1, 0:max(predictedClass), categoryNames);
        % Compute confusion matrix
        size(YVal)
        size(categoricalPred)
        C = confusionmat(YVal, categoricalPred)
        
        % Display confusion chart
        figure
        confusionchart(YVal, categoricalPred,'RowSummary','row-normalized');
        title('Confusion Matrix');
        
        
        precision = diag(C) ./ sum(C, 1)';
        recall = diag(C) ./ sum(C, 2);
        f1Score = 2 * (precision .* recall) ./ (precision + recall);
        
        
        % Display the results
        disp(['Results for Model ' num2str(index0) ' on dataset3 version ' num2str(index3)]);
        disp('Class   Precision   Recall   F1 Score');
        disp([transpose(1:size(C, 1)), precision, recall, f1Score]);
        
        % Save confusion matrix for the current model
        modelDir = fullfile(gatheredDataDir, sprintf('model_%d', index0));
        confusionMatrixDir = fullfile(modelDir, sprintf('dataset3_%d', index3), 'confusion_matrix');
        if ~exist(confusionMatrixDir, 'dir')
            mkdir(confusionMatrixDir);
        end
        save(fullfile(confusionMatrixDir, 'confusion_matrix.mat'), 'C');
        
        f1Score = f1Score(1:4, :);
        f1Score = f1Score';
        modelsdataset3meanF1matrix(index0, index3) = mean(f1Score);
        allF1scoresdataset3(index0, index3 * 4 - 3:index3 * 4) = f1Score;
    
        
    end
end


% Display the best model information
disp(['Best Model valperf: ' num2str(bestModelIdx)]);
disp(['Best Model F1 Score valperf: ' num2str(bestModelF1)]);
disp("---------------------------------------------------------")   
disp('Best Model valperf Information for Each Class:');
for classIdx = 1:numClasses
    disp(['Class ' num2str(classIdx) ':']);
    disp(['   Best Model vaflperf: ' num2str(bestModelIdxperclass(classIdx))]);
    disp(['   Best Model valferp F1 Score: ' num2str(bestModelF1perclass(classIdx))]);
end
disp("---------------------------------------------------------")
disp("---------------------------------------------------------")
disp("---------------------------------------------------------")

% Find the maximum 

bestModelF1dataset1 = max(modelsdataset1meanF1matrix(:));
[bestModelIdxdataset1, bestModelMultfactordataset1] = find(modelsdataset1meanF1matrix == bestModelF1dataset1);

% Display the results for datset1

disp(['Best combination on dataset1: Best Model: ' num2str(bestModelIdxdataset1),'Best Index:' num2str(bestModelMultfactordataset1)]);
disp(['Best F1 Score on dataset1: ' num2str(bestModelF1dataset1)]);
disp("---------------------------------------------------------")   
disp('Best Model-Index combination for Each Class on dataset1:');

for classIdx = 1:numClasses
    selectedColumns = mod(1:size(allF1scoresdataset1, 2), 4) == classIdx;
    subsetMatrix = allF1scoresdataset1(:, selectedColumns);
    maxValue = max(subsetMatrix(:));
    [rowIdx, colIdx] = find(subsetMatrix == maxValue);
    colIdx = find(selectedColumns, colIdx);
    disp(['Class ' num2str(classIdx) ':']);
    disp(['Best combination on dataset1: Best Model ' num2str(rowIdx),'Best Index:' num2str(colIdx)]);
    disp(['Best class F1 Score on dataset1: ' num2str(maxValue)]);
end

disp("---------------------------------------------------------")
disp("---------------------------------------------------------")
disp("---------------------------------------------------------")

% Find the maximum for dataset2
bestModelF1dataset2 = max(modelsdataset2meanF1matrix(:));
[bestModelIdxdataset2, bestModelMultfactordataset2] = find(modelsdataset2meanF1matrix == bestModelF1dataset2);

% Display results for dataset2
disp(['Best combination on dataset2: Best Model: ' num2str(bestModelIdxdataset2), ' Best Index: ' num2str(bestModelMultfactordataset2)]);
disp(['Best F1 Score on dataset2: ' num2str(bestModelF1dataset2)]);
disp("---------------------------------------------------------")   
disp('Best Model-Index combination for Each Class on dataset2:');

for classIdx = 1:numClasses
    selectedColumns = mod(1:size(allF1scoresdataset2, 2), 4) == classIdx;
    subsetMatrix = allF1scoresdataset2(:, selectedColumns);
    maxValue = max(subsetMatrix(:));
    [rowIdx, colIdx] = find(subsetMatrix == maxValue);
    colIdx = find(selectedColumns, colIdx);
    disp(['Class ' num2str(classIdx) ':']);
    disp(['Best combination on dataset2: Best Model ' num2str(rowIdx), ' Best Index: ' num2str(colIdx)]);
    disp(['Best class F1 Score on dataset2: ' num2str(maxValue)]);
end

disp("---------------------------------------------------------")
disp("---------------------------------------------------------")
disp("---------------------------------------------------------")

% Find the maximum for dataset3
bestModelF1dataset3 = max(modelsdataset3meanF1matrix(:));
[bestModelIdxdataset3, bestModelMultfactordataset3] = find(modelsdataset3meanF1matrix == bestModelF1dataset3);

% Display results for dataset3
disp(['Best combination on dataset3: Best Model: ' num2str(bestModelIdxdataset3), ' Best Index: ' num2str(bestModelMultfactordataset3)]);
disp(['Best F1 Score on dataset3: ' num2str(bestModelF1dataset3)]);
disp("---------------------------------------------------------")   
disp('Best Model-Index combination for Each Class on dataset3:');

for classIdx = 1:numClasses
    selectedColumns = mod(1:size(allF1scoresdataset3, 2), 4) == classIdx;
    subsetMatrix = allF1scoresdataset3(:, selectedColumns);
    maxValue = max(subsetMatrix(:));
    [rowIdx, colIdx] = find(subsetMatrix == maxValue);
    colIdx = find(selectedColumns, colIdx);
    disp(['Class ' num2str(classIdx) ':']);
    disp(['Best combination on dataset3: Best Model ' num2str(rowIdx), ' Best Index: ' num2str(colIdx)]);
    disp(['Best class F1 Score on dataset3: ' num2str(maxValue)]);
end



% Save validation
allF1ScoresperclassDir = fullfile(gatheredDataDir, 'allF1Scoresperclass.mat');
save(allF1ScoresperclassDir, 'allF1Scoresperclass');
allF1ScoresDir = fullfile(gatheredDataDir, 'allF1Scores.mat');
save(allF1ScoresDir, 'allF1Scores');

% Save for dataset 1
allF1ScoresperclassDirdataset1 = fullfile(gatheredDataDir, 'allF1scoresdataset1.mat');
save(allF1ScoresperclassDirdataset1, 'allF1scoresdataset1');
allF1ScoresDirdataset1 = fullfile(gatheredDataDir, 'modelsdataset1meanF1matrix.mat');
save(allF1ScoresDirdataset1, 'modelsdataset1meanF1matrix');

% Save for dataset2
allF1ScoresperclassDirdataset2 = fullfile(gatheredDataDir, 'allF1scoresdataset2.mat');
save(allF1ScoresperclassDirdataset2, 'allF1scoresdataset2');
allF1ScoresDirdataset2 = fullfile(gatheredDataDir, 'modelsdataset2meanF1matrix.mat');
save(allF1ScoresDirdataset2, 'modelsdataset2meanF1matrix');

% Save for dataset3
allF1ScoresperclassDirdataset3 = fullfile(gatheredDataDir, 'allF1scoresdataset3.mat');
save(allF1ScoresperclassDirdataset3, 'allF1scoresdataset3');
allF1ScoresDirdataset3 = fullfile(gatheredDataDir, 'modelsdataset3meanF1matrix.mat');
save(allF1ScoresDirdataset3, 'modelsdataset3meanF1matrix');
