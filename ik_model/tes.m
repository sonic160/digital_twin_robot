% Assuming XTrain is your cell array
newXTrain = cell(size(XTrain));

for i = 1:numel(XTrain)
    % Keep the first 500 columns of each matrix
    newXTrain{i} = XTrain{i}(:, 1:500);
end
