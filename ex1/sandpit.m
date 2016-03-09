%% Initialization
clear ; close all; clc

%% ======================= Part 2: Plotting =======================
fprintf('Plotting Data ...\n')
data = load('ex1data1.txt');
X = data(:, 1); y = data(:, 2);
m = length(y); % number of training examples

% Plot Data
% Note: You have to complete the code in plotData.m
plotData(X, y);

%% =================== Part 3: Gradient descent ===================
fprintf('Running Gradient Descent ...\n')

X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters

% Some gradient descent settings
iterations = 1500;
alpha = 0.01;

%% 
% KZ

h = X * theta;

E = h - y;

sqE = E .* E;

SSE = sum(sqE);

J = 1/(2*m) * SSE


%% 


% compute and display initial cost
computeCost(X, y, theta)

%% 

% run gradient descent
theta = gradientDescent(X, y, theta, alpha, iterations);

% print theta to screen
fprintf('Theta found by gradient descent: ');
fprintf('%f %f \n', theta(1), theta(2));

%% figuring out grad descent loop

% Initialize some useful values
m = length(y); % number of training examples
num_iters = 5;
theta = zeros(2, 1);
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    J_history(iter) = computeCost(X, y, theta);
    temp(1) = theta(1) - alpha * J_history(iter);
    temp(2) = theta(2) - alpha * J_history(iter);
    theta = temp
    
end

%% loops

for g = 1:8
    k(g,1) = g+10;
end

%% 
m =length(y);
num_iters = 5;
theta = zeros(2, 1);
J_history = zeros(num_iters, 1);

%% 

h = X * theta;
E = h - y;
sum(E)
E' * X

%% 

iter = 8;

J_history(iter) = computeCost(X, y, theta)


h = X * theta;
E = h - y;
E' * X;
temp(1,1) = theta(1) - alpha * 1/m * sum(E);
temp(2,1) = theta(2) - alpha * 1/m * sum(E' * X);
    
theta = temp

% temp(1,1) = theta(1) - alpha * 1/m * sum ((X * theta) - y);
%% with loop

num_iters = 1500;

for iter = 1:num_iters
    J_history(iter) = computeCost(X, y, theta);
    h = X * theta;
    E = h - y;
    E' * X;
    temp(1,1) = theta(1) - alpha * 1/m * sum(E);
    temp(2,1) = theta(2) - alpha * 1/m * sum(E' * X); 
    theta = temp;
end

plot(1:num_iters, J_history)

% plot(1:num_iters, J_history, 'r')






