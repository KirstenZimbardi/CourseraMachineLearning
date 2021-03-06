function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1; %changed to c in loop
sigma = 0.3; %changed to s in loop

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]';
sigma_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]';

pred_error = zeros(length(C_vec), length(sigma_vec));

for c = 1:length(C_vec)
    for s = 1:length(sigma_vec)
        model = svmTrain(X, y, C_vec(c), @(x1, x2) gaussianKernel(x1, x2, sigma_vec(s)));
        predictions = svmPredict(model, Xval);
        pred_error(c,s) = mean(double(predictions ~= yval));
    end
end

[smallest_error, index] = min(pred_error(:));

j = fix(index/size(pred_error,1));
i = rem(index,size(pred_error,1));
if i > 0
    j = j+1;
end
if i == 0
    i = i+size(pred_error,1);
end
    
%pred_error(i,j)
%pred_error

C = C_vec(i)
sigma = sigma_vec(j)



% =========================================================================

end
