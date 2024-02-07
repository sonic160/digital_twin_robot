
%%%%%%Preprocessing Pipeline

ExecutionEnvironment="gpu";

time_series_length = 1000;
num_classes = 13;

numHiddenUnits = 64;

% Parameters
%struc=load('cellArray2000_circle_line_interpolatesshapes.mat');
struc = load('cellArray600_circle_line_interpolation_motor123error00010203_moy600.mat');
%CD = struc.CD;

cellArray = struc.cellArray;


sizearray = size(cellArray);

numSeq = sizearray(1);   



pattern = mod(0:numSeq-1, num_classes);
categoricalSequence = categorical(pattern, 0:num_classes-1);
totalElements = numel(categoricalSequence);
indexToKeep = round(0.8 * totalElements);

totalCells = numel(cellArray);
index = round(0.8 * totalCells);
XTrain = cellArray(1:index);
XVal = cellArray(index:totalCells);
YTrain = categoricalSequence(1:indexToKeep);
YVal =categoricalSequence(indexToKeep:totalElements);
YVal = renamecats(YVal, '0', '13');
YVal = double(YVal);
SiameseRef = cellArray(1:index);

%%%%%%New code for case of siamese predictions

% instancesPerClass = 1; %designed for 1 atm to protect my sanity - may work on upscaling and averaging later
% % Create a matrix for XVal
% XValtrunc = XVal(1:num_classes*instancesPerClass);
% SiameseRef = reshape(XValtrunc, 13, 3);
% SiameseRefDouble = cellfun(@double, SiameseRef, 'UniformOutput', false);
% 
% % Convert the cell array to a numeric array
% numericArray = cell2mat(SiameseRefDouble.');
% 
% % Convert numeric array to a dlarray
% dlarrayData = dlarray(numericArray, 'CT');
% 
% % Transfer dlarray to GPU (assuming you have a GPU available)
% gpuArrayData = gpuArray(dlarrayData);
% 
% % Display the GPU array if needed
% disp(gpuArrayData);






%%%%%%%

% Extract the first num_classes cells and concatenate them into a 3D array
simpleArray = cat(3, XVal{1:num_classes*4});

% Convert the simple array to a dlarray with 'CTB' dimension label
dlArray = dlarray(simpleArray, 'CTB');

% If training on a GPU, convert data to gpuArray
if strcmp(ExecutionEnvironment, 'gpu')
    SiameseRef = gpuArray(dlArray);
end


%%%%%%%


% Shuffle training data
numTrainSamples = numel(XTrain);
shuffledIndicesTrain = randperm(numTrainSamples);
XTrain = XTrain(shuffledIndicesTrain);
YTrain = YTrain(shuffledIndicesTrain);
% 
 % XTrain = XTrain(1:200)
 % YTrain = YTrain(1:200)
% Shuffle validation data
 numValSamples = numel(XVal);
% shuffledIndicesVal = randperm(numValSamples);
% XVal = XVal(shuffledIndicesVal);
% YVal = YVal(shuffledIndicesVal);

%%

%%%%%Network definitions



featureExtractionNetwork = [
    sequenceInputLayer(6, 'Name', 'inputFEN')
    bilstmLayer(numHiddenUnits, 'OutputMode', 'sequence')
    bilstmLayer(numHiddenUnits, 'OutputMode', 'sequence')
    convolution1dLayer(2,5,'Stride',2,'Padding',1)
    maxPooling1dLayer(2,'Stride',3,'Padding',1)
];


ExtractionNetwork = dlnetwork(featureExtractionNetwork);


%{ previous version, testing new alternative below}


% relationshipMeasurementNetwork = [
%     sequenceInputLayer(10, 'Name', 'inputMeN')
%     convolution1dLayer(5, 16, 'Padding', 'same', 'Stride', 2, 'Name', '1D_CNN_1rmn')
%     convolution1dLayer(5, 16, 'Padding', 'same', 'Stride', 1, 'Name', '1D_CNN_2rmn')
%     globalAveragePooling1dLayer('Name', 'GlobalAveragePoolingrmn')
%     fullyConnectedLayer(1, 'Name', 'FullyConnectedrmn')
% ];


relationshipMeasurementNetwork = [
    sequenceInputLayer(10, 'Name', 'inputMeN')
    convolution1dLayer(5, 5, 'Padding', 'same', 'Stride', 2, 'Name', '1D_CNN_1rmn')
    reluLayer
    maxPooling1dLayer(1,Stride=2)
    reluLayer
    convolution1dLayer(5, 5, 'Padding', 'same', 'Stride', 1, 'Name', '1D_CNN_2rmn')
    reluLayer
    globalAveragePooling1dLayer('Name', 'GlobalAveragePoolingrmn')
    fullyConnectedLayer(1, 'Name', 'FullyConnectedrmn')
];








MeasurementNetwork = dlnetwork(relationshipMeasurementNetwork);

faultClassificationNetwork = [
    sequenceInputLayer(5, 'Name', 'inputFCN')
    convolution1dLayer(5, 32, 'Padding', 'same', 'Stride', 2)
    globalAveragePooling1dLayer('Name', 'GlobalAveragePoolingfcn')
    fullyConnectedLayer(num_classes)
    softmaxLayer
    %classificationLayer %%%may need to be removed from model and performed externally
];

ClassificationNetwork = dlnetwork(faultClassificationNetwork);



%%%%%Training Options
numIterations = 17999;
miniBatchSize = 150;
learningRate = 1.2e-3;
gradDecay = 0.9;
gradDecaySq = 0.99;
bestLoss = inf;

%%%%% Further Configuration




trailingAvgSubnetex = [];
trailingAvgSqSubnetex = [];
trailingAvgSubnetclas = [];
trailingAvgSqSubnetclas = [];
trailingAvgSubnetmes = [];
trailingAvgSqSubnetmes = [];


ExecutionEnvironment = "gpu";
monitor = trainingProgressMonitor(Metrics = ["Loss", "Classification_Accuracy", "Measurement_Accuracy", "Siamese_Accuracy"], XLabel = "Iteration", Info = "ExecutionEnvironment");

if canUseGPU
    gpu = gpuDevice;
    disp(gpu.Name + " GPU detected and available for training.")
end

if canUseGPU
    updateInfo(monitor, ExecutionEnvironment = gpu.Name + " GPU")
else
    updateInfo(monitor, ExecutionEnvironment = "CPU")
end

%%%%% Custom training loop, to be altered

sstart = tic;
iteration = 0;
saved = 0;
validationFrequency = 250;  % Set the frequency for validation accuracy calculation

bestLoss = inf;
bestSetEx = [];
bestSetMes = [];
bestSetClas = [];

% Loop over mini-batches.
while iteration < numIterations && ~monitor.Stop

    iteration = iteration + 1;
    if learningRate > 1e-5
     learningRate = learningRate * 0.99974;
    else
        learningRate = 1e-5;
    end

    % Extract mini-batch of image pairs and pair labels
    [X1, X2, pairLabels] = getTwinBatch(XTrain, YTrain, miniBatchSize);

    % Convert mini-batch of data to dlarray. Specify the dimension labels
    % "CTB" (channel, spatial, batch) for image data
    X1 = dlarray(X1, "CTB");
    X2 = dlarray(X2, "CTB");

    % If training on a GPU, then convert data to gpuArray.
    if ExecutionEnvironment == "gpu"
        X1 = gpuArray(X1);
        X2 = gpuArray(X2);
    end

    % Evaluate the model loss and gradients using dlfeval and the model Loss
    [loss, mesgradientsSubnet, clasgradientsSubnet, exgradientsSubnet] = dlfeval(@modelLoss, ExtractionNetwork, MeasurementNetwork, ClassificationNetwork, X1, X2, pairLabels, SiameseRef);

    % Update the feature extraction subnetwork parameters.
    [ExtractionNetwork, trailingAvgSubnetex, trailingAvgSqSubnetex] = adamupdate(ExtractionNetwork, exgradientsSubnet, trailingAvgSubnetex, trailingAvgSqSubnetex, iteration, learningRate, gradDecay, gradDecaySq);

    % Update the measurement network parameters.
    [MeasurementNetwork, trailingAvgSubnetmes, trailingAvgSqSubnetmes] = adamupdate(MeasurementNetwork, mesgradientsSubnet, trailingAvgSubnetmes, trailingAvgSqSubnetmes, iteration, learningRate, gradDecay, gradDecaySq);

    % Update the classification network parameters.
    [ClassificationNetwork, trailingAvgSubnetclas, trailingAvgSqSubnetclas] = adamupdate(ClassificationNetwork, clasgradientsSubnet, trailingAvgSubnetclas, trailingAvgSqSubnetclas, iteration, learningRate, gradDecay, gradDecaySq);

    % Calculate validation accuracy - method 1 every 500 iterations
    if mod(iteration, validationFrequency) == 0
        reduced_Array = XVal(1:399);
        reshapedArray = permute(cell2mat(reduced_Array), [2, 3, 1]);
        reshapedArray = cat(3, reduced_Array{:});
        reduced_dlarray = dlarray(reshapedArray, "CTB");
        intermediate_array = predict(ExtractionNetwork, reduced_dlarray);
        final_array = predict(ClassificationNetwork, intermediate_array);
        [~, maxIndices] = max(final_array);
        resultArray = maxIndices;
        comparisonArray = resultArray ==  YVal(1:399);
        accuracy = sum(comparisonArray) / numel(comparisonArray);

        
        SiameseRefPred = predict(ExtractionNetwork, SiameseRef);
        final_array_siamese = zeros(size(resultArray));
        for i = 1:length(intermediate_array)
            best_pred = 0;
            best_index = 0;
            for j = 1:num_classes
                counter = 0;
                sigmoid_prediction = 0;
                while counter <= 3
                    
                    concatenated_dlarray = cat(1, intermediate_array(:, i, :), SiameseRefPred(:, j + num_classes*counter, :));
                    prediction = predict(MeasurementNetwork, concatenated_dlarray);
                    sigmoid_prediction = sigmoid(prediction) + sigmoid_prediction;
                    counter = counter + 1;
                end
                    if sigmoid_prediction > best_pred
                        best_pred = sigmoid_prediction;
                        best_index = j;
                    end
                
            end
            final_array_siamese(i) = best_index;
        end
        siamesecomparisonarray = final_array_siamese == YVal(1:399);
        siamese_accuracy = sum(siamesecomparisonarray) / numel(siamesecomparisonarray);




measurement_accuracy = 0;
for i = 1:num_classes*4
    for j = 1:num_classes*4
        % Skip self-comparisons
        if i == j
            continue;
        end

        % Concatenate pairs of SiameseRefPred elements
        concatenated_dlarray = cat(1, SiameseRefPred(:, i, :), SiameseRefPred(:, j, :));

        % Predict using MeasurementNetwork
        prediction = predict(MeasurementNetwork, concatenated_dlarray);

        % Determine the expected label based on the condition
        expected_label = mod(i, num_classes) == mod(j, num_classes);

        % Calculate sigmoid prediction
        sigmoid_prediction = sigmoid(prediction);

        % Check if prediction is correct based on the condition
        is_correct = (expected_label && sigmoid_prediction >= 0.5) || (~expected_label && sigmoid_prediction < 0.5);

        % Accumulate correct predictions
        measurement_accuracy = measurement_accuracy + is_correct;
    end
end

% Calculate the rate of correctly predicted pairs
measurement_accuracy = measurement_accuracy / (num_classes*4 *(num_classes-1)*4);









        % Update the best model if the current loss is better
        if loss < bestLoss
            bestLoss = loss;
            bestSetEx = ExtractionNetwork;
            bestSetMes = MeasurementNetwork;
            bestSetClas = ClassificationNetwork;
        end

        % Update the training progress monitor.
        recordMetrics(monitor, iteration, Classification_Accuracy=accuracy);
        monitor.Progress = 100 * iteration / numIterations;
        recordMetrics(monitor, iteration, Measurement_Accuracy=measurement_accuracy);
        monitor.Progress = 100 * iteration / numIterations;   
        recordMetrics(monitor, iteration, Siamese_Accuracy=siamese_accuracy);
        monitor.Progress = 100 * iteration / numIterations;   

    else
        % Update the training progress monitor without accuracy information
        recordMetrics(monitor, iteration, Loss=loss);
        monitor.Progress = 100 * iteration / numIterations;
    end

end

%%%%% The following lines serve to evaluate accuracy -could be called
%%%%% during training loop to compute validation and implement auto-cutoff



%% Uncomment ABOVE FOR TRAINING

accuracymes = zeros(1,5);
accuracyclas1 = zeros(1,5);
accuracyclas2 = zeros(1,5);
accuracyBatchSize = 200; %the more different classes there are, the more important it is for this to be large

    for i = 1:1
        % Extract mini-batch of image pairs and pair labels
    [X1,X2,pairLabelsAcc] = getTwinBatch(XVal, YVal,accuracyBatchSize);
    
        % Convert mini-batch of data to dlarray. Specify the dimension labels
        % "SSCB" (spatial, spatial, channel, batch) for image data - various
        % combinations to be considered 
        X1 = dlarray(X1,"CTB");
        X2 = dlarray(X2,"CTB");
    
        % If training on a GPU, then convert data to gpuArray.
        if  ExecutionEnvironment == "gpu"
            X1 = gpuArray(X1);
            X2 = gpuArray(X2);
        end
    
        % Evaluate predictions using trained network
        [Y,Y1,Y2] = predictTwin(ExtractionNetwork,MeasurementNetwork, ClassificationNetwork,X1,X2);
    
        % Convert predictions to binary 0 or 1 or to classes
        Y = gather(extractdata(Y));
        Y1 = gather(extractdata(Y1));
        Y2 = gather(extractdata(Y2));
    
        Y = round(Y);
        Y1 = round(Y1);
        Y2 = round(Y2);
    
        categoricalseries1 = categorical(pairLabelsAcc(1,:));
    categoricalseries2 = categorical(pairLabelsAcc(2,:));
    size(categoricalseries1')
    categoricalseries1 = onehotencode(categoricalseries1',2);
    categoricalseries2 = onehotencode(categoricalseries2',2);
    
    
    
    Y1 = Y1';
    Y2 = Y2';
    disp(Y)
    disp(pairLabelsAcc(3,:))
    precision = sum(Y & pairLabelsAcc(3,:)) / sum(Y);
    recall = sum(Y & pairLabelsAcc(3,:)) / sum(pairLabelsAcc(3,:));
    
    F1_score = 2 * (precision * recall) / (precision + recall);
    averageF1Score = mean(F1_score);



    % Compute average accuracy for the minibatch


    accuracymes(i) = sum(Y == pairLabelsAcc(3,:))/accuracyBatchSize;
    equal1 = sum(Y1 == categoricalseries1, 2);
    equal2 = sum(Y2 == categoricalseries2, 2);
    accuracyclas1(i) = sum(equal1== size(categoricalseries1,2))/accuracyBatchSize;
    accuracyclas2(i) = sum(equal2== size(categoricalseries2,2))/accuracyBatchSize;



averageAccuracyMes = mean(accuracymes)*100;
averageAccuracyClas = (mean(accuracyclas1)*100 +  mean(accuracyclas2)*100)/2;
disp(['Average Accuracy (Mes): ', num2str(averageAccuracyMes), '%']);
disp(['Average Accuracy (Clas): ', num2str(averageAccuracyClas), '%']);
disp(['Average F1 Score: ', num2str(averageF1Score)]);




disp(['Average Accuracy (Mes): ', num2str(averageAccuracyMes), '%']);
disp(['Average Accuracy (Clas): ', num2str(averageAccuracyClas), '%']);
disp(['Average F1 Score: ', num2str(averageF1Score)]);
    %testing bins for siamese display
falsePairings = Y == pairLabelsAcc(3,:);
list0 = {};
list1 = {};

% Iterate over pairLabelsAcc columns
% for colIdx = 1:numel(falsePairings)
%     pairLabels = pairLabelsAcc(1:2, colIdx);
% 
%     % Check the corresponding element in falsePairings
%     if falsePairings(colIdx) == 0
% 
%         if pairLabels(2) < pairLabels(1)
%             stringRepresentation = sprintf('%d%d', pairLabels(2), pairLabels(1));
%         else
%             stringRepresentation = sprintf('%d%d', pairLabels(1), pairLabels(2));
%         end
%         list0{end+1} = stringRepresentation;
%     elseif falsePairings(colIdx) == 1
% 
%         if pairLabels(2) < pairLabels(1)
%             stringRepresentation = sprintf('%d%d', pairLabels(2), pairLabels(1));
%         else
%             stringRepresentation = sprintf('%d%d', pairLabels(1), pairLabels(2));
%         end
%         list1{end+1} = stringRepresentation;
%     end
% end
% 

%% 
reduced_Array = XVal(1:400);
reshapedArray = permute(cell2mat(reduced_Array), [2, 3, 1]);
reshapedArray = cat(3, reduced_Array{:});
reduced_dlarray = dlarray(reshapedArray,"CTB");
intermediate_array = predict(ExtractionNetwork, reduced_dlarray);
final_array = predict(ClassificationNetwork, intermediate_array);
[~, maxIndices] = max(final_array);
resultArray = maxIndices;
comparisonArray = resultArray == mod(99 + (1:numel(resultArray)), 4);
sum(comparisonArray)/numel(comparisonArray);

end


%% This segment here does some testing on the measurement network. Can be adpated to make predictions

SiameseRef = predict(ExtractionNetwork, SiameseRef);
% Initialize the results table
resultsTable = zeros(4, 4);

% Iterate over all possible pairs of integers between 1 and 13
for i = 1:num_classes
    for j = num_classes
        % Create concatenated_dlarray for the current pair
        concatenated_dlarray = cat(1, SiameseRef(:, i, :), SiameseRef(:, j, :));

        % Run predictions using the MeasurementNetwork
        predictions = predict(MeasurementNetwork, concatenated_dlarray);

        % Apply sigmoid function to the predictions
        sigmoid_predictions = sigmoid(predictions);

        % Round the sigmoid predictions to 0 or 1
        rounded_predictions = round(sigmoid_predictions);

        % Store the rounded result in the table
        resultsTable(i, j) = rounded_predictions;

        % Optionally, you can display intermediate results if needed
        disp(['Pair (' num2str(i) ', ' num2str(j) '): ' num2str(rounded_predictions)]);
    end
end

% Display the final results table
disp('Results Table:')
disp(resultsTable);


%%

% Initialize variables for tracking the maximum similarity
simpleArray = cat(3, XVal{14:39});

% Convert the simple array to a dlarray with 'CTB' dimension label
dlArray = dlarray(simpleArray, 'CTB');

% If training on a GPU, convert data to gpuArray
if strcmp(ExecutionEnvironment, 'gpu')
    ElementSource = gpuArray(dlArray);
end




ElementSource = predict(ExtractionNetwork,ElementSource);
for evaluated_index = 1:8
    max_similarity = -Inf;
most_similar_index = 0;
    evaluated_element = ElementSource(:, evaluated_index, :);
% Iterate over all original elements
for original_index = 1:4
    % Create concatenated_dlarray for the current pair
    concatenated_dlarray = cat(1, SiameseRef(:, original_index, :), evaluated_element);

    % Run predictions using the MeasurementNetwork
    predictions = predict(MeasurementNetwork, concatenated_dlarray);

    % Apply sigmoid function to the predictions
    sigmoid_predictions = sigmoid(predictions);

    % Round the sigmoid predictions to 0 or 1
    rounded_predictions = round(sigmoid_predictions);

    % Calculate similarity (you can use other similarity measures as needed)
    similarity = sum(rounded_predictions(:) == 1) / numel(rounded_predictions);

    % Update maximum similarity if the current pair is more similar
    if similarity > max_similarity
        max_similarity = similarity;
        most_similar_index = original_index;
    end

    
end

% Display the index of the most similar original element
disp(['Most Similar Element Index: ' num2str(most_similar_index)]);

end
%% 


%%%%%Helper functions


function [Y,Y1,Y2] = forwardTwin(featureExtractionNetwork,MeasurementNetwork, ClassificationNetwork,X1,X2)
% forwardTwin accepts the network and pair of training images, and
% returns a prediction of the probability of the pair being similar (closer
% to 1) or dissimilar (closer to 0). Use forwardTwin during training, as
% well as predicted classes for both

% Pass the first image through the twin subnetwork

Y1 = forward(featureExtractionNetwork,X1);


% Pass the second image through the twin subnetwork
Y2 = forward(featureExtractionNetwork,X2);


% concatenate for distance measurement

Y = cat(1,Y1,Y2);


% Pass the result through the measrement network
Y = forward(MeasurementNetwork, Y);


% Convert to probability between 0 and 1.
Y = sigmoid(Y);

%Proceed to classify both samples independantly - need to check how it
%works without clazssification layer

Y1 = forward(ClassificationNetwork,Y1);
Y2 = forward(ClassificationNetwork,Y2);




end


function [Y,Y1,Y2] = forwardTwinSiamese(featureExtractionNetwork,MeasurementNetwork, SiameseRef,X1,X2)

Y1 = forward(featureExtractionNetwork,X1);

Y2 = forward(featureExtractionNetwork,X2);


Y = cat(1,Y1,Y2);

Y = forward(MeasurementNetwork, Y);

Y = sigmoid(Y);
extracted_ref = forward(featureExtractionNetwork,SiameseRef);


% %Proceed to classify both samples independantly - need to check how it
% %works without clazssification layer
%     bestPred1 = 1;
%     bestPred2 = 2;
%     best_sum1 = 0;
%     best_sum2 = 0;
% 
% for i = 1:length(SiameseRef)
%         sum1 = 0;
%         sum2 = 0;
% 
%         for j = 1:width(SiameseRef)
% 
%             % to be adapted for usage on vectors
%             Y1_conc = cat(1,Y1,SiameseRef(j,i));
%             Y2_conc = cat(1,SiameseRef(j,i),Y2);
%             Y1_conc = forward(MeasurementNetwork, Y1_conc);
%             Y2_conc = forward(MeasurementNetwork, Y2_conc);
%             Y1_conc = sigmoid(Y1);
%             Y2_conc = sigmoid(Y2);
%             sum1 = sum1 + Y1_conc;
%             sum2 = sum2 + Y2_conc;
%         end
% 
%         if sum1 > best_sum1
%             bestPred1 = i;
%             best_sum1 = sum1;
%         end
%         if sum2 > best_sum2
%             bestPred2 = i;
%             best_sum2 = sum2;
%         end


% Assuming Y1 is a cell array containing the batch and extracted_ref is a dlarray
% Assuming MeasurementNetwork is the network used for prediction

numElementsInBatch = size(Y1, 2);
num_classes =13; % Assuming extracted_ref is a dlarray

% Initialize predictions array
predictions1 = zeros(1, numElementsInBatch);
predictions2 = zeros(1, numElementsInBatch);

for i = 1:numElementsInBatch
   
    currentBatchElement1 = Y1(:, i, :);
    currentBatchElement2 = Y2(:, i, :);


    
    % Initialize similarity scores for the current batch element
    similarityScores1 = zeros(1, num_classes);
    
    for j = 1:num_classes
        % Concatenate the current batch element with the j-th sample


        concatenatedData1 = cat(1, currentBatchElement1, extracted_ref(:,j,:));
        concatenatedData2 = cat(1, extracted_ref(:,j,:), currentBatchElement2);
        % Pass the concatenated data through MeasurementNetwork
        prediction1 = forward(MeasurementNetwork, concatenatedData1);
        prediction2 = forward(MeasurementNetwork, concatenatedData2);
        % Apply sigmoid activation
        prediction1 = sigmoid(prediction1);
        prediction2 = sigmoid(prediction2);

        % Calculate a similarity score (e.g., L2 distance, cosine similarity, etc.)
        similarityScores1(j) =  prediction1;
        similarityScores2(j) = prediction2;
    end

    % Find the index of the sample with the highest similarity
    [~, maxIndex1] = max(similarityScores1);
    [~, maxIndex2] = max(similarityScores2);
    % Assign the maxIndex to the corresponding element of Y1
    predictions1(i) = maxIndex1;
    predictions2(i) = maxIndex2;


   
end



%note that unlike before the prediction is int, not onehot -will require
%further changes in code


Y1 =  predictions1;
Y2 = predictions2;
size(Y1)

end


function  [Y,Y1,Y2] = predictTwin(featureExtractionNetwork,MeasurementNetwork, ClassificationNetwork,X1,X2)
% predictTwin accepts the network and pair of images, and returns a
% prediction of the probability of the pair being similar (closer to 1) or
% dissimilar (closer to 0). Use predictTwin during prediction.

% Pass the first image through the twin subnetwork
Y1 = predict(featureExtractionNetwork,X1);


% Pass the second image through the twin subnetwork
Y2 = predict(featureExtractionNetwork,X2);


% concatenate for distance measurement
Y = cat(1,Y1,Y2);

% Pass the result through the measrement network
Y = predict(MeasurementNetwork, Y);

% Convert to probability between 0 and 1.
Y = sigmoid(Y);

%Proceed to classify both samples independantly - need to check how it
%works without clazssification layer

Y1 = predict(ClassificationNetwork,Y1);
Y2 = predict(ClassificationNetwork,Y2);


end


function [loss,mesgradientsSubnet, clasgradientsSubnet, exgradientsSubnet] = modelLoss(extractionNet,measurementNet, classificationNet,X1,X2,pairLabels,SiameseRef)

% Pass the image pair through the network to obtain predicted classes 
 [Y,Y1,Y2] = forwardTwin(extractionNet,measurementNet, classificationNet,X1,X2);
%[Y,Y1,Y2] = forwardTwinSiamese(extractionNet,measurementNet, SiameseRef,X1,X2);
% Calculate binary cross-entropy loss - first we one-hot our classes
categoricalseries1 = categorical(pairLabels(1,:));
categoricalseries2 = categorical(pairLabels(2,:));



%commenter les deux lignes suivantes en cas 
categoricalseries1 = onehotencode(categoricalseries1',2);
categoricalseries2 = onehotencode(categoricalseries2',2);
% 
% Y1 = categorical(Y1);
% disp(Y1)
% Y2 = categorical(Y2);
% Y1 = onehotencode(Y1',2);
% Y2 = onehotencode(Y2',2);
% 
% disp(Y1)

%weights to penalise worst class confusions


weights = ones(1, size(categoricalseries1, 2));



classification1loss = crossentropy(Y1,categoricalseries1', weights,'WeightsFormat','UC', ClassificationMode="multilabel");
classification2loss = crossentropy(Y2,categoricalseries2', weights,'WeightsFormat','UC', ClassificationMode="multilabel");
measurementloss = crossentropy(Y,pairLabels(3,:), ClassificationMode="multilabel");


%% uncomment in non-siamese

% measurementloss = crossentropy(Y,pairLabels(3,:), ClassificationMode="multilabel");
% 
% categoricalseries1 = double(categoricalseries1);
% Y1 = double(Y1);
% Y2 = double(Y2);
% categoricalseries2 = double(categoricalseries2);
% classification1loss =   mean((double(bsxfun(@eq, categoricalseries1, unique(categoricalseries1)')) - Y1).^2, 'all');
% classification2loss =  mean((double(bsxfun(@eq, categoricalseries2, unique(categoricalseries2)')) - Y2).^2, 'all');
%cpnsider doing solmething cool here, like increasing loss if the class
%predictions agree and measurement doesn't or something

% Calculate gradients of the loss with respect to the network learnable
% parameters.
% measurementloss;
% classification1loss;
% classification2loss;
% 
loss = classification1loss + classification2loss + measurementloss;
%loss =  measurementloss ;  %temp measure to look at the siamese part in particular


[mesgradientsSubnet] = dlgradient(measurementloss,measurementNet.Learnables); 
[clasgradientsSubnet] = dlgradient(classification1loss + classification2loss,classificationNet.Learnables); 
[exgradientsSubnet] = dlgradient(classification1loss + classification2loss + measurementloss,extractionNet.Learnables); 



% classification1loss = classification1loss/20
% classification2loss = classification2loss/20
% 
% classification1loss = gpuArray(dlarray(single(classification1loss)))
% classification2loss = gpuArray(dlarray(single(classification2loss)))
% loss = classification1loss + classification2loss + measurementloss
% 
% [mesgradientsSubnet] = dlgradient(measurementloss,measurementNet.Learnables); 
% [clasgradientsSubnet] = dlgradient(measurementloss,classificationNet.Learnables); 
% [exgradientsSubnet] = dlgradient(classification1loss + classification2loss + measurementloss,extractionNet.Learnables); 

end

% % % The following is juts causing problems with the whole
% 'higherorderderivative' shtick

% % % % % Use dlfeval to run the function and trace operations
% % % [loss, exgradientsSubnet, mesgradientsSubnet] = dlfeval(@computeLossAndGradients, classification1loss, classification2loss, measurementloss, classificationNet, extractionNet, measurementNet);
% % % 
% % % %TODO : understand why it seems gradientparams is not needed here
% % % end
% % % 
% % % function [loss, exgradientsSubnet, mesgradientsSubnet] = computeLossAndGradients(class1loss, class2loss, measloss, clasNet, extrNet, measNet)
% % %     % Forward pass
% % %     disp("the function is entered")
% % %     loss = class1loss + class2loss + measloss;
% % % 
% % %     % Backward pass
% % % 
% % %     [exgradientsSubnet] = dlgradient(class1loss + class2loss + measloss, extrNet.Learnables, 'EnableHigherDerivatives',true);
% % %     [mesgradientsSubnet] = dlgradient(measloss, measNet.Learnables, 'EnableHigherDerivatives',true);
% % % 
% % % end




function [X1, X2, pairLabels] = getTwinBatch(X, Y, miniBatchSize)
    % Initialize the output.
    pairLabels = zeros(3, miniBatchSize);
    imgSize = [size(X{1}, 1), size(X{1}, 2)]; 
    usedClasses = categorical([]);

    X1 = zeros([imgSize, miniBatchSize], 'single');
    X2 = zeros([imgSize, miniBatchSize], 'single');

    % Create a batch containing similar and dissimilar pairs of images.
    for i = 1:miniBatchSize
        
       
        choice = rand(1);

        % Randomly select a similar or dissimilar pair of images.
        if choice < 0.5
            [pairIdx1, pairIdx2, pairLabels(3,i), usedClasses] = getSimilarPair(Y, usedClasses);
        else
            [pairIdx1, pairIdx2, pairLabels(3,i), usedClasses] = getDissimilarPair(Y, usedClasses);
        end
  
        X1(:, :, i) = X{pairIdx1}; 
        X2(:, :, i) = X{pairIdx2}; 
        pairLabels(1,i) = Y(pairIdx1);
        pairLabels(2,i) = Y(pairIdx2);
    end
end

function [pairIdx1, pairIdx2, pairLabel, usedClasses] = getSimilarPair(classLabel, usedClasses)
    % Find all unique classes not in usedClasses.
    classes = setdiff(categories(classLabel), usedClasses);

    % If all classes have been used at least once, reset the list.
    if isempty(classes)
        usedClasses = categorical([]);
        classes = categories(classLabel);
    end

    % Choose a class randomly from the remaining classes.
    classChoice = randi(numel(classes));

    % Add the chosen class to the list.
    usedClasses = vertcat(usedClasses, classes(classChoice));

    % Find the indices of all the observations from the chosen class.
    idxs = find(classLabel == classes(classChoice));

    % Randomly choose two different images from the chosen class.
    pairIdxChoice = randperm(numel(idxs), 2);
    pairIdx1 = idxs(pairIdxChoice(1));
    pairIdx2 = idxs(pairIdxChoice(2));
    pairLabel = 1;
end

function [pairIdx1, pairIdx2, label, usedClasses] = getDissimilarPair(classLabel, usedClasses)
    % Find all unique classes not in usedClasses.
    classes = setdiff(categories(classLabel), usedClasses);

    %%not perfect, because adds both classes to "explored" when either list
    %%only recieves 1 - howver the odds of this coming up should be
    %%(1/2)^300

    % If all classes have been used at least once or there are fewer than
    % two remaining classes, reset the list.
    if isempty(classes) || numel(classes) < 2
        usedClasses = categorical([]);
        classes = categories(classLabel);
    end

    % Choose two different classes randomly from the remaining classes.
    classesChoice = randperm(numel(classes), 2);

    % Add the chosen classes to the list.
    usedClasses = vertcat(usedClasses, classes(classesChoice(1)), classes(classesChoice(2)));

    % Find the indices of all the observations from the first and second
    % classes.
    idxs1 = find(classLabel == classes(classesChoice(1)));
    idxs2 = find(classLabel == classes(classesChoice(2)));

    % Randomly choose one image from each class.
    pairIdx1Choice = randi(numel(idxs1));
    pairIdx2Choice = randi(numel(idxs2));
    pairIdx1 = idxs1(pairIdx1Choice);
    pairIdx2 = idxs2(pairIdx2Choice);
    label = 0;
end


function [Y] = predictSingleSample(featureExtractionNetwork, ClassificationNetwork, X)
    % Convert input to dlarray. Specify the dimension labels
    X = dlarray(X, 'CTB');

    % If training on a GPU, convert data to gpuArray
    if  ExecutionEnvironment == "gpu"
        X = gpuArray(X);
    end

    % Pass the image through the twin subnetwork
    Y = predict(featureExtractionNetwork, X);

    % Proceed to classify the sample using the classification network
    Y = predict(ClassificationNetwork, Y);
end



function predictions = predictWithMeasurement(featureExtractionNetwork, MeasurementNetwork, classMatrix, X)
    % Pass the input image through the feature extraction network
    X = predict(featureExtractionNetwork, X);

    % Initialize predictions array
    num_classes = size(classMatrix, 1);
    numSamples = size(classMatrix, 2);
    predictions = zeros(num_classes, numSamples);

    % Process each class separately
    for classIdx = 1:num_classes
        % Repeat the processed input for each sample in the class
        repeatedX = repmat(X, [1, numSamples]);

        % Concatenate the processed input with each sample in the class
        combinedInputs = cat(1, repeatedX, classMatrix(classIdx, :));

        % Calculate resemblance for each combination
        resemblances = predict(MeasurementNetwork, combinedInputs);

        % Reshape the resemblances array for each sample in the class
        resemblances = reshape(resemblances, [numSamples, numSamples + 1]);

        % Calculate the average resemblance for each sample
        avgResemblances = mean(resemblances, 2);

        % Assign the maximum average resemblance as the prediction for each sample
        predictions(classIdx, :) = avgResemblances;
    end
end
