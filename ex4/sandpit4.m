%sandpit for ex4
%working out nnCostFx

% Preset from ex4.m - comment out when running effectively            
lambda = 1;

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
J_unreg = (1/m) * Em
%% adding regularisation
%Theta1

Theta1_squared = Theta1(:,2:size(Theta1,2)) .* Theta1(:,2:size(Theta1,2));
Theta1_SS = sum(Theta1_squared);
Theta1_SS = sum(Theta1_SS,2);

Theta2_squared = Theta2(:,2:size(Theta2,2)) .* Theta2(:,2:size(Theta2,2));
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

%% Sigmoid function

z = [-15:-10;10:15;0:5]



gz = 1.0 ./ (1.0 + exp(-z))

g = gz .* (1 - gz)



                                   