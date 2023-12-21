CB=load('cellArray500interpolatesshapes.mat');

% Get the current directory
currentDir = pwd;

% Construct the full path to the file in the parent directory
%filePathInParent = fullfile(currentDir, '..', 'lstmv3_2bilayers_500interpolate_200hiddenunnit_dropout0_2_alr_128batch.mat');

% Load the file
%net = load(filePathInParent);
strucnet=load('lstmv3_2bilayers_153_line_circle_interpolate_200hiddenunnit_dropout0_2_alr_128batch.mat');
net=strucnet.net;
cellArray=CB.cellArray;
sizearray = size(cellArray);
numSeq = sizearray(1); % Number of sequences
disp(numSeq)

numClasses = 13;  % Number of classes

% Generate categorical sequence

pattern = mod(0:numSeq-1, numClasses);
categoricalSequence = categorical(pattern, 0:numClasses-1);
%disp(categoricalSequence);


XVal = cellArray;
YVal =categoricalSequence;

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
% ne pas uncomment le demon des plots svp

% % Identify misclassified samples
% misclassifiedIndices = find(YVal ~= categoricalPred);
% 
% % Analyze misclassifications
% for i = 1:length(misclassifiedIndices)
%     index = misclassifiedIndices(i);
%     disp(['Misclassified sample at index ', num2str(index)]);
%     % Further analysis of misclassified samples, e.g., plot the sequence:
%     figure
%     plot(XVal{index}')
%     xlabel("Time Step")
%     title(['Misclassified Sample ', num2str(index)])
% end

