function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y


% You need to return the following variables correctly 
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% Initialize some useful values
m = length(y); % number of training examples

% breaking cost function into simple steps

h = X * theta;

E = h - y;

sqE = E .* E;

SSE = sum(sqE);

J = 1/(2*m) * SSE;


% or compunded into 1 step NB harder to read and debug
% J = 1/(2* (length(y))) * sum( ((X * theta) - y) .* ((X * theta) - y) );

% partial for grad descent
% (X * theta) - y

% =========================================================================

end
