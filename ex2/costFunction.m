function [J, grad] = costFunction(thetaT, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
%m = length(y); % number of training examples
[m, n] = size(X);

%alpha = 0.05;
%theta = zeros(1,l(2));

% You need to return the following variables correctly 

z = X * thetaT

h = sigmoid(z)

E = (-y .* log(h)) - ((1-y) .* log(1-h))

J = 1/m * sum(E)

grad = zeros(size(thetaT))

Egrad = h - y

for j = 1:n
    Error(j,:) = Egrad .* X(:,j);
    grad(j,:) = 1/m * sum(Error(j,:));
end
    
%grad

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%








% =============================================================

end
