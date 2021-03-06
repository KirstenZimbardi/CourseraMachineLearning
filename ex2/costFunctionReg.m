function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
%m = length(y); % number of training examples
[m, n] = size(X);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

z = X * theta;

h = sigmoid(z);

E = (-y .* log(h)) - ((1-y) .* log(1-h));

reg = sum(theta(2:n) .* theta(2:n));

J = 1/m * sum(E) + (lambda/(2*m) * reg);

Egrad = h - y;


for j = 1:n
    Error(j,:) = Egrad .* X(:,j);
    grad(j,:) = 1/m * sum(Error(j,:));
end

for j = 2:n
    grad(j,:) = grad(j,:) + ((lambda/m) * theta(j));
end

% =============================================================

end
