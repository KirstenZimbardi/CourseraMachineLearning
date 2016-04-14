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

% random initialisation
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

            
% adding X0            
[m, n] = size(X);
X = [ones(m, 1) X];
[m, n] = size(X);

% converting y to matrix where columns are outcome classes
for k = 1:num_labels
    c(:,k) = y == k;
end

%forward layer 1 -> 2
z2 = X * initial_Theta1';
a2 = sigmoid(z2);
a2 = [ones(m,1) a2];
%forward layer 2 -> 3
z3 = a2 * initial_Theta2';
h = sigmoid(z3);

%cost
J = (1/m) * (sum(sum((c .* log(h)) + ((1-c) .* log(h)),2),1));
    
%errors
d3 = h - c;
d2 = d3 * initial_Theta2(:,2:end) .* sigmoidGradient(z2);

%gradients and scaling
Delta1 = d2' * X;
Theta1_grad = (1/m) * Delta1;
    
Delta2 = d3' * a2;
Theta2_grad = (1/m) * Delta2;


%unrolling
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
