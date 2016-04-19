function [theta] = trainLinearReg(X, y, lambda)
%TRAINLINEARREG Trains linear regression given a dataset (X, y) and a
%regularization parameter lambda
%   [theta] = TRAINLINEARREG (X, y, lambda) trains linear regression using
%   the dataset (X, y) and regularization parameter lambda. Returns the
%   trained parameters theta.
%

[m, n] = size(X);
% add X0 IF  needed 
if (X(1:m,1) ~= ones(m,1)) 
   X = [ones(m, 1) X];
   [m, n] = size(X);
end     


% Initialize Theta
initial_theta = zeros(n, 1); 


% Create "short hand" for the cost function to be minimized
%costFunction = @(t) linearRegCostFunction([ones(m,1) X], y, t, lambda);
costFunction = @(t) linearRegCostFunction(X, y, t, lambda);


% Now, costFunction is a function that takes in only one argument
options = optimset('MaxIter', 200, 'GradObj', 'on');

% Minimize using fmincg
theta = fmincg(costFunction, initial_theta, options);

end
