%resetting data set
%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

% Load Training Data
fprintf('Loading Data ...\n')

load('ex4data1.mat');
m = size(X, 1);

%% ================ Part 5: Sigmoid Gradient  ================
%  Before you start implementing the neural network, you will first
%  implement the gradient for the sigmoid function. You should complete the
%  code in the sigmoidGradient.m file.
%

fprintf('\nEvaluating sigmoid gradient...\n')

g = sigmoidGradient([1 -0.5 0 0.5 1]);
fprintf('Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]:\n  ');
fprintf('%f ', g);
fprintf('\n\n');

%% ================ Part 6: Initializing Pameters ================
%  In this part of the exercise, you will be starting to implment a two
%  layer neural network that classifies digits. You will start by
%  implementing a function to initialize the weights of the neural network
%  (randInitializeWeights.m)

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

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
a2 = ones(m,hidden_layer_size+1);
D2 = zeros(num_labels, hidden_layer_size);

for k = 1:num_labels
    c(:,k) = y == k;
end

%% for loop
for t = 1:m
    %forward layer 1 -> 2
    z(t,:) = X(t,:) * iT1t;
    %forward layer 2 -> 3
    a2(t,2:end) = sigmoid(z(t,:));
    z2(t,:) = a2(t,:) * iT2t;
    h(t,:) = sigmoid(z2(t,:));
    %error in layer 3
    d3(t,:) = h(t,:) - c(t,:);
    %back prop layer 3 -> 2
    %d2(t,:) = h(t,:) .* (1-h(t,:)); %old version
    d3t = d3';
    d2 = d3t(:,t) * a2(t,2:end);
    D2 = D2 + d2;
end

%% gradient
Theta2_grad = d3' * a2;
%D2g = D2(:,2:end);
Theta1_grad = D2 * X;
grad = [Theta1_grad(:) ; Theta2_grad(:)];

J = (1/m) * sum(grad);




%% =============== Part 7: Implement Backpropagation ===============
%  Once your cost matches up with ours, you should proceed to implement the
%  backpropagation algorithm for the neural network. You should add to the
%  code you've written in nnCostFunction.m to return the partial
%  derivatives of the parameters.
%
fprintf('\nChecking Backpropagation... \n');

%  Check gradients by running checkNNGradients
checkNNGradients;

%fprintf('\nProgram paused. Press enter to continue.\n');
%pause;


%% 
