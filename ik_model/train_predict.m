
load('lstm_test.mat', 'untrainedNetwork');
load('trainingData.mat', 'trainingData', 'trainingLabels');

net = myCustomNetwork();  % Define your network function

% Specify training options
options = trainingOptions('adam', ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 32, ...
    'ValidationData', {validationData, validationLabels}, ...
    'ValidationFrequency', 10, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% Train the network
net = trainNetwork(trainingData, trainingLabels, net, options);