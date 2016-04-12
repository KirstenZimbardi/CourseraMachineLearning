%clipboard
nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

if ~exist('theta', 'var') || isempty(theta)
    theta = nn_params;
end
