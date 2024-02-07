
loadedcell=load('./cellArray2000_circle_line_interpolatesshapes.mat');

%cellArray=loadedcell.cellArray;
cellArray=loadedcell.CD.cellArray;
resultCellArray=zeros(500,1);

for i = 1:500
    % Check if i is divisible by 4

    % Transpose the matrix
    transposedMatrix = cellArray{i*4}';

    % Perform termwise subtractions using columns 1 through 6
    resultMatrix1 = abs(abs(transposedMatrix(:, 1)) - abs(transposedMatrix(:, 3)));
    resultMatrix2 = abs(abs(transposedMatrix(:, 2)) - abs(transposedMatrix(:, 4)));
    resultMatrix3 = abs(abs(transposedMatrix(:, 3)) - abs(transposedMatrix(:, 6)));

    % Compute the average value of each resulting column
    averageValue1 = mean(resultMatrix1);
    averageValue2 = mean(resultMatrix2);
    averageValue3 = mean(resultMatrix3);

    % Compute the average of the averageValues 1 through 3
    averageOfAverages = mean([averageValue1, averageValue2, averageValue3]);

    % Store the result in the new cell array
    resultCellArray(i) = averageOfAverages>0.036;
    
end

% Display the results
disp('Average of averageValues 1 through 3 for each matrix:');
disp(resultCellArray);


