%sandpit for ex4
%working out nnCostFx

% Preset from ex4.m - comment out when running effectively            
lambda = 0;

%fx parameters
%function [J grad] = nnCostFunction(nn_params, ...
                                   %input_layer_size, ...
                                   %hidden_layer_size, ...
                                   %num_labels, ...
                                   %X, y, lambda)
                                   
%% 
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


%% 
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


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
J = (1/m) * Em



%% 

                                   