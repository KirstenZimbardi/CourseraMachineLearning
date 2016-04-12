% new sandpit for backpropogation inside nnCostFunction
% original fx
%Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
%                 hidden_layer_size, (input_layer_size + 1));

%Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
%                 num_labels, (hidden_layer_size + 1));
%nn_params = [Theta1(:) ; Theta2(:)];

% improved initialisation
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

             
% adding X0            
[m, n] = size(X);
X = [ones(m, 1) X];
[m, n] = size(X);

%% test for loop

iT1t = initial_Theta1';
iT2t = initial_Theta2';
a2 = ones(m,hidden_layer_size+1);

for k = 1:num_labels
    c(:,k) = y == k;
end

for t = 1:m
    %forward layer 1 -> 2
    z(t,:) = X(t,:) * iT1t;
    %forward layer 2 -> 3
    a2(t,2:end) = sigmoid(z(t,:));
    z2(t,:) = a2(t,:) * iT2t;
    h(t,:) = sigmoid(z2(t,:));
    %back prop layer 3 ->
    d3(t,:) = h(t,:) - c(t,:);
    d2(t,:) = h(t,:) .* (1-h(t,:)); %old version
end

%gradient
Theta2_grad = d3' * a2;
a2g = a2(:,2:end);
Theta1_grad = a2g' * X;
grad = [Theta1_grad(:) ; Theta2_grad(:)];

D = (1/m) * grad;

%% 
% error for layer 2
% new version not working 
%sg = sigmoidGradient(z2); 
%d2a = d3 * Theta2;
%d2 =d2a .* sg;
% no error for layer 1 since that is the data input

%% 



%% cost
% J = grad/m ie 1/m * grad



