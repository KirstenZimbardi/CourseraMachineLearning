function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%
         
% You need to return the following variables correctly 
%J = 0;
%Theta1_grad = zeros(size(initial_Theta1));
%Theta2_grad = zeros(size(initial_Theta2));

% ====================== YOUR CODE HERE ======================

%re-roll the unrolled random initial Thetas from nn_params
initial_Theta1 = reshape(nn_params(1:(hidden_layer_size * (input_layer_size+1))),[hidden_layer_size,(input_layer_size+1)]);
initial_Theta2 = reshape(nn_params((hidden_layer_size * (input_layer_size+1))+1:end),[num_labels, (hidden_layer_size+1)]);

% add X0            
[m, n] = size(X);
X = [ones(m, 1) X];
[m, n] = size(X);

% convert y vector to c matrix where columns are outcome classes (oneVsAll
% format)
for k = 1:num_labels
    c(:,k) = y == k;
end

% Neural Network

%forward layer 1 -> 2
z2 = X * initial_Theta1';
a2 = sigmoid(z2);
a2 = [ones(m,1) a2];
%forward layer 2 -> 3
z3 = a2 * initial_Theta2';
h = sigmoid(z3);

%errors
d3 = (h - c);
d2 = d3 * initial_Theta2(:,2:end) .* sigmoidGradient(z2);

%gradients 
Delta1 = d2' * X;
Delta2 = d3' * a2;

%scaling
Theta1_grad = (1/m) * Delta1;
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) ...
    + ((lambda/m) * initial_Theta1(:,2:end));

Theta2_grad = (1/m) * Delta2;
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) ...
    + ((lambda/m) * initial_Theta2(:,2:end));


%unrolling
grad = [Theta1_grad(:) ; Theta2_grad(:)];

%cost
E = (-c .* log(h)) - ((1-c) .* log(1-h));
ssE = sum(sum(E,2),1);

ssT1 = sum(sum(initial_Theta1(:,2:end) .^2,2),1);
ssT2 = sum(sum(initial_Theta2(:,2:end) .^2,2),1);
reg = ssT1 + ssT2;

J = (1/m * ssE) + (lambda/(2*m) * reg);

end
