%bits out of NNN and to speed up checks

% setup variables
if ~exist('lambda', 'var') || isempty(lambda)
    lambda = 0;
end

input_layer_size = 3;
hidden_layer_size = 5;
num_labels = 3;
m = 5;

% We generate some 'random' test data
Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size);
Theta2 = debugInitializeWeights(num_labels, hidden_layer_size);
% Reusing debugInitializeWeights to generate X
X  = debugInitializeWeights(m, input_layer_size - 1);
y  = 1 + mod(1:m, num_labels)';

% Unroll parameters
nn_params = [Theta1(:) ; Theta2(:)];

%% Short hand for cost function
costFunc = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, ...
                               num_labels, X, y, lambda);
%% run nnCostFunction
[cost, grad] = costFunc(nn_params);

%% run nnCostFunction manually

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


%% run numgrad
numgrad = computeNumericalGradient(costFunc, nn_params);

%% run numgrad manually

J = costFunc;
theta = nn_params;

numgrad = zeros(size(theta));
perturb = zeros(size(theta));
e = 1e-4;
for p = 1:numel(theta)
    % Set perturbation vector
    perturb(p) = e;
    loss1 = J(theta - perturb);
    loss2 = J(theta + perturb);
    % Compute Numerical Gradient
    numgrad(p) = (loss2 - loss1) / (2*e);
    perturb(p) = 0;
end


%% compare
disp([numgrad grad]);
diff = norm(numgrad-grad)/norm(numgrad+grad)