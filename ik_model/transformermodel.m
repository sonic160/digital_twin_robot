%%%%%%Preprocessing Pipeline

ExecutionEnvironment="gpu";

time_series_length = 1000;
num_classes = 13;
numClasses = num_classes;
numChannels = 6; %1000 pour longueur de message? Replace with your desired number of channels
numHeads = 8;     % Replace with your desired number of heads
numKeys = 64;
numWords= 128; %(subdivisions de la sequence)
sequenceLength = time_series_length; % mettre len_timeseries
maxpos=sequenceLength+1;
numOut=num_classes;
miniBatchSize=64;
classes = categorical({'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'});
% Parameters
struc=load('cellArray600_circle_line_interpolation_motor123error00010203_moy600.mat');
%CD = struc.CD;
cellArray = struc.cellArray;
%cellArray = struc.CD.cellArray;


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
%YTrain=num2cell(YTrain);
YVal =categoricalSequence(indexToKeep:totalElements);
%YVal = num2cell(YVal);
% 
% % Convert cell arrays to sequences
% XTrainSeq = cellfun(@(x) cell2mat(x), XTrain, 'UniformOutput', false);
% 
% % Convert categorical responses to sequences
% YTrainSeq = sequenceData(YTrain);
% 
% % Convert cell arrays to sequences for validation data
% XValSeq = cellfun(@(x) cell2mat(x), XVal, 'UniformOutput', false);
% 
% % Convert categorical responses to sequences for validation data
% YValSeq = sequenceData(YVal);

tower = [
    sequenceInputLayer(numChannels, 'Name', 'tower-input')
    %%%il y a les deux tours entre les couches ici
    concatenationLayer(1,2,'Name','concat-1');
    fullyConnectedLayer(numOut,'Name', 'FCC-h1-CS');
    softmaxLayer('Name','soft-mx-g1')
    multiplicationLayer(2,'Name','mult-C')

    concatenationLayer(1,2,'Name','concat-2');
    fullyConnectedLayer(numOut,'Name', 'FCC-h2-CS');
    softmaxLayer('Name','soft-mx-g2')
    multiplicationLayer(2,'Name','mult-S')
    concatenationLayer(1,2,'Name','concat-y');
    fullyConnectedLayer(numOut,'Name', 'FCC-out');
    softmaxLayer('Name','soft-out');
    %classificationLayer('Name','output');
    classificationLayer('Name','output','Classes', classes);
];

tower1stepwiseencoder = [
    positionEmbeddingLayer(numChannels,maxpos,Name="pos-emb_1");
    %sinusoidalPositionEncodingLayer(numChannels, 'Name', 'sin-pos-enc');
    sinusoidalPositionEncodingLayer(numChannels, 'Name', 'sin-pos-enc','Positions', 'temporal-indices');
    fullyConnectedLayer(numChannels, 'Name', 'FCC-replace-word-emb-1');
    tanhLayer('Name', 'tanh-1');
    additionLayer(2, 'Name', 'add1');
];
lgraph = layerGraph(tower);
lgraph = addLayers(lgraph, fullyConnectedLayer(numOut,'Name', 'FCC-t1-out'));
lgraph = addLayers(lgraph, tanhLayer('Name',"tanh-out-1") );
lgraph = addLayers(lgraph, fullyConnectedLayer(numOut,'Name', 'FCC-t2-out'));
lgraph = addLayers(lgraph, tanhLayer('Name',"tanh-out-2") );

% Connect tower1 to the first transformer block
lgraph = disconnectLayers(lgraph,'tower-input','concat-1/in1');
lgraph = disconnectLayers(lgraph,'mult-C','concat-2/in1');
lgraph = connectLayers(lgraph, "tanh-out-1", 'mult-C/in2');
lgraph = connectLayers(lgraph, "tanh-out-2", 'mult-S/in2');
lgraph = connectLayers(lgraph, 'mult-C', 'concat-y/in2');

lgraph = addLayers(lgraph, tower1stepwiseencoder);
%lgraph = connectLayers(lgraph, 'tower-input', 'sin-pos-enc');
lgraph = connectLayers(lgraph, 'tower-input', "pos-emb_1");
lgraph = disconnectLayers(lgraph,"sin-pos-enc","FCC-replace-word-emb-1");
lgraph = connectLayers(lgraph,"pos-emb_1","FCC-replace-word-emb-1");
%lgraph = connectLayers(lgraph,"sin-pos-enc","add1/in2");
lgraph = connectLayers(lgraph,'sin-pos-enc',"add1/in2");

% Connect tower2 to the first transformer block
tower2embedder = [
    positionEmbeddingLayer(numChannels,maxpos,Name="pos-emb_2");
    fullyConnectedLayer(numChannels,Name="FCC-replace-word-emb-2_2")%replaces word embedding layer as we do not have to map words to numbers
    tanhLayer('Name','tanh-2_2')
    %autre possibilité learned positional imbedding
    
    ];

% Connect tower2 to the first transformer block
lgraph = addLayers(lgraph, tower2embedder);
lgraph = connectLayers(lgraph, 'tower-input', "pos-emb_2");

%transformerblocks

% Create an array to store different versions of transformerBlock
transformerBlocks = cell(1, 12);

% Define the custom transformer blocks
for i = 1:12
    transformerBlocks{i} = [
        %causal attention mask for tower 1
        selfAttentionLayer(numHeads, numKeys, 'Name', sprintf('self-att_%d', i),'AttentionMask','causal');
        additionLayer(2, 'Name', sprintf('add2_%d', i));
        layerNormalizationLayer('Name', sprintf('lay-norm1_%d', i));
        fullyConnectedLayer(numChannels, 'Name', sprintf('fc1_%d', i));
        reluLayer('Name', sprintf('relu1_%d', i));
        fullyConnectedLayer(numChannels, 'Name', sprintf('fc2_%d', i));
        additionLayer(2, 'Name', sprintf('add3_%d', i));
        layerNormalizationLayer('Name', sprintf('lay-norm2_%d', i));
    ];
end

%Tower 1
lgraph = addLayers(lgraph, transformerBlocks{1});
lgraph = connectLayers(lgraph, 'add1', 'self-att_1');
lgraph = connectLayers(lgraph, 'add1', sprintf('add2_%d/in2', 1));
lgraph = connectLayers(lgraph, sprintf('lay-norm1_%d', 1), sprintf('add3_%d/in2', 1));

% Add six transformer blocks in series
for i = 2:6
    % Add the transformer block to the layer graph
    lgraph = addLayers(lgraph, transformerBlocks{i});
    
    % Connect the transformer blocks in series
    lgraph = connectLayers(lgraph, sprintf('lay-norm2_%d', i-1), sprintf('self-att_%d', i));
    lgraph = connectLayers(lgraph, sprintf('lay-norm2_%d', i-1), sprintf('add2_%d/in2', i));
    lgraph = connectLayers(lgraph, sprintf('lay-norm1_%d', i), sprintf('add3_%d/in2', i));
end



lgraph = connectLayers(lgraph, sprintf('lay-norm2_%d', 6), 'FCC-t1-out');

lgraph = connectLayers(lgraph,'FCC-t1-out', "tanh-out-1");
lgraph = connectLayers(lgraph,"tanh-out-1",'concat-1/in1');
lgraph = connectLayers(lgraph,"tanh-out-1",'concat-2/in1');

%Tower 2
lgraph = addLayers(lgraph, transformerBlocks{7});
lgraph = connectLayers(lgraph, 'tanh-2_2', 'self-att_7');
lgraph = connectLayers(lgraph, 'tanh-2_2', sprintf('add2_%d/in2', 7));
lgraph = connectLayers(lgraph, sprintf('lay-norm1_%d', 7), sprintf('add3_%d/in2', 7));

% Add six transformer blocks in series
for i = 8:12
    % Add the transformer block to the layer graph
    lgraph = addLayers(lgraph, transformerBlocks{i});
    
    % Connect the transformer blocks in series
    lgraph = connectLayers(lgraph, sprintf('lay-norm2_%d', i-1), sprintf('self-att_%d', i));
    lgraph = connectLayers(lgraph, sprintf('lay-norm2_%d', i-1), sprintf('add2_%d/in2', i));
    lgraph = connectLayers(lgraph, sprintf('lay-norm1_%d', i), sprintf('add3_%d/in2', i));
end



lgraph = connectLayers(lgraph, sprintf('lay-norm2_%d', 12), 'FCC-t2-out');

lgraph = connectLayers(lgraph,'FCC-t2-out', "tanh-out-2");
lgraph = connectLayers(lgraph,"tanh-out-2",'concat-1/in2');
lgraph = connectLayers(lgraph,"tanh-out-2",'concat-2/in2');
% Visualize the layer graph
figure
plot(lgraph)

options = trainingOptions("adam", ...
    ExecutionEnvironment="gpu", ...
    GradientThreshold=1, ...
    MaxEpochs=500, ...
    MiniBatchSize=miniBatchSize, ...
    ValidationData={XVal,YVal}, ... %new
    ValidationFrequency=20, ...     %new
    SequenceLength="longest", ...
    L2Regularization = 0.0001, ...  %new
    Shuffle="once", ...   
    Verbose=0, ...
    Plots="training-progress");

net = trainNetwork(XTrain,YTrain,lgraph,options);
%net = trainNetwork(XTrain,YTrain,net.Layers,options);
%net = trainNetwork(XTrainSeq, YTrainSeq, net.Layers, options);



save('transformer2_c_l_i_600_13_motorerror00010203_0123_full_1000_150hiddenunnit_dropout0_2_alr_64_batch.mat','net')
% Make predictions on the validation set
YPred = predict(net, XVal);

% Find the column index of the maximum probability for each row
[~, predictedClass] = max(YPred, [], 2);

% Create a categorical array from the predicted class indices
categoryNames = cellstr(num2str((0:max(predictedClass))'));  % Assuming classes are 0-based
categoricalPred = categorical(predictedClass - 1, 0:max(predictedClass), categoryNames);

% Compute confusion matrix
C = confusionmat(YVal, categoricalPred)


% Connect the last transformer block to the final layers
%lgraph = connectLayers(lgraph, sprintf('TransformerBlock_%d/lay-norm2', 6), 'add2/in2');

