function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1); % initialise cost matrix

% values in function that could be included but in function can be altered
% theta = zeros(2, 1); % initialise theta
% num_iters = 1500; % number of iterantions to run

for iter = 1:num_iters
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    J_history(iter) = computeCost(X, y, theta);
    h = X * theta;
    E = h - y;
    E1 = E .* X(:,2);
    temp(1,1) = theta(1,1) - alpha * 1/m * sum(E);
    temp(2,1) = theta(2,1) - alpha * 1/m * sum(E1); 
    theta = temp;
    
end

plot(1:num_iters, J_history, 'r')


