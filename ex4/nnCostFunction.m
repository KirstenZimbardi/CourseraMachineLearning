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

%Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
%                 hidden_layer_size, (input_layer_size + 1));

%Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
%                 num_labels, (hidden_layer_size + 1));


%% nnCostFunction

% random initialisation
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

            
% adding X0            
[m, n] = size(X);
X = [ones(m, 1) X];
[m, n] = size(X);

iT1t = initial_Theta1';
iT2t = initial_Theta2';
%iT1t = Theta1';
%iT2t = Theta2';
%set up empty variables for the for loop
z2 = zeros(m,hidden_layer_size);
a2 = ones(m,hidden_layer_size+1);
z3 = zeros(m,num_labels);
h = zeros(m,num_labels);
d3 = zeros(1,num_labels);
Delta3 = zeros(1,num_labels);
d2 = zeros(1,hidden_layer_size+1);
Delta2 = zeros(1, hidden_layer_size+1);

for k = 1:num_labels
    c(:,k) = y == k;
end

%% for loop
%for t = 1:m
    %forward layer 1 -> 2
    z2(t,:) = X(t,:) * iT1t;
    %forward layer 2 -> 3
    a2(t,2:end) = sigmoid(z2(t,:));
    z3(t,:) = a2(t,:) * iT2t;
    h(t,:) = sigmoid(z3(t,:));
    %error in layer 3
    %d3(t,:) = h(t,:) - c(t,:); %prev version - not accumulating error over
    %iterations because new error going into new row each time - try
    %accumulating error instead:
    d3 = h(t,:) - c(t,:);
    Delta3 = Delta3 + d3;
    %back prop layer 3 -> 2
    %d2(t,:) = h(t,:) .* (1-h(t,:)); %very old version
    %d3t = d3'; %prev version
    %d2 = d3t(:,t) * a2(t,2:end); %prev version
    %new version of d2 accumulating error for each training eg
    %d2 = (theta2' * d3) .* (a2 .* (1-a2))
    d21 = d3 * initial_Theta2;
    d22 = a2(t,:) .* (1 - a2(t,:));
    d2 = d21 .* d22;
    Delta2 = Delta2 + d2;
%end

%% D (ie partial derivative of Jtheta)
D2 = (1/m) * Delta2;
D3 = (1/m) * Delta2;


%% gradient
Theta2_grad = d3' * a2;
%D2g = D2(:,2:end);
Theta1_grad = Delta2 * X;
grad = [Theta1_grad(:) ; Theta2_grad(:)];

J = (1/m) * sum(grad);

%% 

         
% You need to return the following variables correctly 
%J = 0;
%Theta1_grad = zeros(size(Theta1));
%Theta2_grad = zeros(size(Theta2));

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



end
