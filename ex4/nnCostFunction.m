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

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

            
[m, n] = size(X);

% Add ones to the X data matrix
X = [ones(m, 1) X];
[m, n] = size(X);
            
% Setup some useful variables
%m = size(X, 1);

%% g function
% layer 1 -> 2
z = X * Theta1';
a2 = sigmoid(z);

%% g function
% layer 2 ->3
a2 = [ones(m, 1) a2];
z2 = a2 * Theta2';
h = sigmoid(z2);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



%% Cost

for k = 1:num_labels
    c = y == k;
    E(k,:) = (-c .* log(h(:,k))) - ((1-c) .* log(1-h(:,k)));
end

%% sum for 1:k classes
Ek = sum(E);

%% sum for 1:m training examples
Em = sum(Ek,2);

%%  * 1/m
J_unreg = (1/m) * Em

%% adding regularisation
%Theta1

Theta1_squared = Theta1(:,2:end) .* Theta1(:,2:end);
Theta1_SS = sum(Theta1_squared);
Theta1_SS = sum(Theta1_SS,2);

Theta2_squared = Theta2(:,2:end) .* Theta2(:,2:end);
Theta2_SS = sum(Theta2_squared);
Theta2_SS = sum(Theta2_SS,2);

J = J_unreg + ((lambda/(2*m)) * (Theta1_SS + Theta2_SS))


%% gradient
for k = 1:num_labels
    c(:,k) = y == k;
end
%% error for layer 3

d3 = h - c;
%% error for layer 2

d2 = h .* (1-h);
% no error for layer 1 since that is the data input
%% getting Theta1_grad and Theta2_grad based on clued size

Theta2_grad = d3' * a2;

a2g = a2(:,2:(hidden_layer_size+1));

Theta1_grad = a2g' * X;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
