function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
%m = length(y); % number of training examples


%theta = initial_theta;

[m, n] = size(X);
% add X0 IF needed 
if (X(1:m,1) ~= ones(m,1)) 
   X = [ones(m, 1) X];
   [m, n] = size(X);
end     
    

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
h = X * theta;

E = h - y;

SSE = sum(E.^2);
%J = 1/(2*m) * SSE;
Jreg = sum(theta(2:end,:).^2);
J = (1/(2*m) * SSE) + (lambda/(2*m) * Jreg);

grad = (1/m) * (X' * E);
gradReg = (lambda/m) * theta(2:end,:);
grad = grad + [zeros(1,1); gradReg];

% =========================================================================

end
