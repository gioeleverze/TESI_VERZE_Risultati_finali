function [InputMatrix,count] = sliding_window(X, windowSize,tau)
[numDims, numSamples] = size(X);  %% 
if numDims ~= 1
    error('Input should be one dimension!');
end
InputMatrix = zeros(windowSize, numSamples);
count = 0;  
for i = 1 : numSamples
    Head = 1 + tau*(i-1);
    End = Head + windowSize-1;
    if End > numSamples
        break;
    end
    x = X(Head : End)';
    InputMatrix(:,i) = x;
    count = count + 1;
end
end

