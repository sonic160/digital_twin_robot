
%%%%%%Preprocessing Pipeline



time_series_length = 1000;
num_classes = 13;
numClasses = num_classes;
numHiddenUnits = 64;

% Parameters
struc=load('cellArray201_circle_line_interpolation_motor123error00010203.mat');
%CD = struc.CD;

cellArray = struc.cellArray;


sizearray = size(cellArray);

numSeq = sizearray(1);   



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



% Shuffle training data
numTrainSamples = numel(XTrain);
shuffledIndicesTrain = randperm(numTrainSamples);
XTrain = XTrain(shuffledIndicesTrain);
YTrain = YTrain(shuffledIndicesTrain);

% Shuffle validation data
numValSamples = numel(XVal);
shuffledIndicesVal = randperm(numValSamples);
XVal = XVal(shuffledIndicesVal);
YVal = YVal(shuffledIndicesVal);

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


relationshipMeasurementNetwork = [
    sequenceInputLayer(10, 'Name', 'inputMeN')
    convolution1dLayer(5, 16, 'Padding', 'same', 'Stride', 2, 'Name', '1D_CNN_1rmn')
    convolution1dLayer(5, 16, 'Padding', 'same', 'Stride', 1, 'Name', '1D_CNN_2rmn')
    globalAveragePooling1dLayer('Name', 'GlobalAveragePoolingrmn')
    fullyConnectedLayer(1, 'Name', 'FullyConnectedrmn')
];


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
    fullyConnectedLayer(numClasses)
    softmaxLayer
    %classificationLayer %%%may need to be removed from model and performed externally
];

ClassificationNetwork = dlnetwork(faultClassificationNetwork);



%%%%%Training Options
numIterations = 8995;
miniBatchSize = 50;
learningRate = 2e-4;
gradDecay = 0.9;
gradDecaySq = 0.99;


%%%%% Further Configuration




trailingAvgSubnetex = [];
trailingAvgSqSubnetex = [];
trailingAvgSubnetclas = [];
trailingAvgSqSubnetclas = [];
trailingAvgSubnetmes = [];
trailingAvgSqSubnetmes = [];


ExecutionEnvironment="gpu";
monitor = trainingProgressMonitor(Metrics="Loss",XLabel="Iteration",Info="ExecutionEnvironment");
if canUseGPU
    gpu = gpuDevice;
    disp(gpu.Name + " GPU detected and available for training.")
end
if canUseGPU
    updateInfo(monitor,ExecutionEnvironment=gpu.Name + " GPU")
else
    updateInfo(monitor,ExecutionEnvironment="CPU")
end




%%%%% Custom training loop, to be altered

start = tic;
iteration = 0;
saved = 0;

% Loop over mini-batches.
while iteration < numIterations && ~monitor.Stop

    iteration = iteration + 1;

    % Extract mini-batch of image pairs and pair labels

    %todo : cell2mat and spatial dimensions fix and dlnetwork conversion to
    %be handled later - best solution probably is batch-by-batch conversion
    [X1,X2,pairLabels] = getTwinBatch(XTrain, YTrain,miniBatchSize);
   
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

    % Evaluate the model loss and gradients using dlfeval and the model Loss
  
    %for measurement, hence the 3rd line
    [loss,mesgradientsSubnet, clasgradientsSubnet, exgradientsSubnet] = dlfeval(@modelLoss,ExtractionNetwork,MeasurementNetwork, ClassificationNetwork,X1,X2,pairLabels);

    %for classification


    % Update the feature extraction subnetwork parameters.
    [ExtractionNetwork,trailingAvgSubnetex,trailingAvgSqSubnetex] = adamupdate(ExtractionNetwork,exgradientsSubnet, ...
        trailingAvgSubnetex,trailingAvgSqSubnetex,iteration,learningRate,gradDecay,gradDecaySq);

    % Update the measurement network parameters.
    [MeasurementNetwork,trailingAvgSubnetmes,trailingAvgSqSubnetmes] = adamupdate(MeasurementNetwork,mesgradientsSubnet, ...
        trailingAvgSubnetmes,trailingAvgSqSubnetmes,iteration,learningRate,gradDecay,gradDecaySq);

        % Update the classification network parameters.
    [ClassificationNetwork,trailingAvgSubnetclas,trailingAvgSqSubnetclas] = adamupdate(ClassificationNetwork,clasgradientsSubnet, ...
        trailingAvgSubnetclas,trailingAvgSqSubnetclas,iteration,learningRate,gradDecay,gradDecaySq);


    % Update the training progress monitor.
    recordMetrics(monitor,iteration,Loss=loss);
    monitor.Progress = 100 * iteration/numIterations;

end
 
if monitor.Stop && saved == 0  %bloc pour savegarder en cas d'interruption manuelle - pas sûr que ça marche comme ça mais bon
    saved =1;
    save('ExtractionNetworkCheckpointmes.mat', 'ExtractionNetwork', 'trailingAvgSubnetex', 'trailingAvgSqSubnetex');
    save('MeasurementNetworkCheckpointmes.mat', 'MeasurementNetwork', 'trailingAvgSubnetmes', 'trailingAvgSqSubnetmes');
    save('ClassificationNetworkCheckpointmes.mat', 'ClassificationNetwork', 'trailingAvgSubnetclas', 'trailingAvgSqSubnetclas');
end


%%%%%% The following lines serve to evaluate accuracy -could be called
%%%%%% during training loop to compute validation and implement auto-cutoff


accuracymes = zeros(1,5);
accuracyclas1 = zeros(1,5);
accuracyclas2 = zeros(1,5);
accuracyBatchSize = 150; %the more different classes there are, the more important it is for this to be large

for i = 1:5
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
categoricalseries1 = onehotencode(categoricalseries1',2);
categoricalseries2 = onehotencode(categoricalseries2',2);


disp(size(categoricalseries1))
disp(size(categoricalseries2))
Y1 = Y1';
Y2 = Y2';

precision = sum(Y & pairLabelsAcc(3,:)) / sum(Y);
recall = sum(Y & pairLabelsAcc(3,:)) / sum(pairLabelsAcc(3,:));

F1_score = 2 * (precision * recall) / (precision + recall);
averageF1Score = mean(F1_score);

disp(['Average Accuracy (Mes): ', num2str(averageAccuracyMes), '%']);
disp(['Average Accuracy (Clas): ', num2str(averageAccuracyClas), '%']);
disp(['Average F1 Score: ', num2str(averageF1Score)]);


    % Compute average accuracy for the minibatch


    accuracymes(i) = sum(Y == pairLabelsAcc(3,:))/accuracyBatchSize;
    equal1 = sum(Y1 == categoricalseries1, 2);
    equal2 = sum(Y2 == categoricalseries2, 2);
    accuracyclas1(i) = sum(equal1== size(categoricalseries1,2))/accuracyBatchSize;
    accuracyclas2(i) = sum(equal2== size(categoricalseries2,2))/accuracyBatchSize;



averageAccuracyMes = mean(accuracymes)*100;
averageAccuracyClas = (mean(accuracyclas1)*100 +  mean(accuracyclas2)*100)/2;
    %testing bins for siamese display
falsePairings = Y == pairLabelsAcc(3,:);
list0 = {};
list1 = {};

% Iterate over pairLabelsAcc columns
for colIdx = 1:numel(falsePairings)
    pairLabels = pairLabelsAcc(1:2, colIdx);

    % Check the corresponding element in falsePairings
    if falsePairings(colIdx) == 0
       
        if pairLabels(2) < pairLabels(1)
            stringRepresentation = sprintf('%d%d', pairLabels(2), pairLabels(1));
        else
            stringRepresentation = sprintf('%d%d', pairLabels(1), pairLabels(2));
        end
        list0{end+1} = stringRepresentation;
    elseif falsePairings(colIdx) == 1
       
        if pairLabels(2) < pairLabels(1)
            stringRepresentation = sprintf('%d%d', pairLabels(2), pairLabels(1));
        else
            stringRepresentation = sprintf('%d%d', pairLabels(1), pairLabels(2));
        end
        list1{end+1} = stringRepresentation;
    end
end







end






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


function [loss,mesgradientsSubnet, clasgradientsSubnet, exgradientsSubnet] = modelLoss(extractionNet,measurementNet, classificationNet,X1,X2,pairLabels)

% Pass the image pair through the network to obtain predicted classes 
 [Y,Y1,Y2] = forwardTwin(extractionNet,measurementNet, classificationNet,X1,X2);

% Calculate binary cross-entropy loss - first we one-hot our classes
categoricalseries1 = categorical(pairLabels(1,:));
categoricalseries2 = categorical(pairLabels(2,:));
categoricalseries1 = onehotencode(categoricalseries1',2);
categoricalseries2 = onehotencode(categoricalseries2',2);
disp(size(categoricalseries1))
disp(size(categoricalseries2))


%weights to penalise worst class confusions


weights = ones(1, size(categoricalseries1, 2));
weights(1) = 1.3; % weight for class 
weights(4) = 1.3; % weight for class 4



classification1loss = crossentropy(Y1,categoricalseries1', weights,'WeightsFormat','UC', ClassificationMode="multilabel");
classification2loss = crossentropy(Y2,categoricalseries2', weights,'WeightsFormat','UC', ClassificationMode="multilabel");
measurementloss = crossentropy(Y,pairLabels(3,:), ClassificationMode="multilabel");

%cpnsider doing solmething cool here, like increasing loss if the class
%predictions agree and measurement doesn't or something

% Calculate gradients of the loss with respect to the network learnable
% parameters.
classification1loss;
classification2loss;
measurementloss;
loss = classification1loss + classification2loss + measurementloss;
%loss =  measurementloss ;  %temp measure to look at the siamese part in particular


[mesgradientsSubnet] = dlgradient(measurementloss,measurementNet.Learnables); 
[clasgradientsSubnet] = dlgradient(classification1loss + classification2loss,classificationNet.Learnables); 
[exgradientsSubnet] = dlgradient(classification1loss + classification2loss + measurementloss,extractionNet.Learnables); 

%TODO : understand why it seems gradientparams is not needed here
end





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
            [pairIdx1, pairIdx2, pairLabels(3,i)] = getSimilarPair(Y, usedClasses);
        else
            [pairIdx1, pairIdx2, pairLabels(3,i)] = getDissimilarPair(Y, usedClasses);
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

    % If all classes have been used at least once, reset the list.
    if isempty(classes)
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

