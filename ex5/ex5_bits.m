%ex5 bits to streamline checks
%% Initialization
clear ; close all; clc

load ('ex5data1.mat');

% m = Number of examples
m = size(X, 1);

%% Unregularised cost function
theta = [1 ; 1];
%J = linearRegCostFunction([ones(m, 1) X], y, theta, 1);
J = linearRegCostFunction(X, y, theta, 1);

fprintf(['Cost at theta = [1 ; 1]: %f '...
         '\n(this value should be about 303.993192)\n'], J);

%% Regularised version     
theta = [1 ; 1];
%[J, grad] = linearRegCostFunction([ones(m, 1) X], y, theta, 1);
[J, grad] = linearRegCostFunction(X, y, theta, 1);


fprintf(['Gradient at theta = [1 ; 1]:  [%f; %f] '...
         '\n(this value should be about [-15.303016; 598.250744])\n'], ...
         grad(1), grad(2));
     
%% Training theta
lambda = 0;
%[theta] = trainLinearReg([ones(m, 1) X], y, lambda);
[theta] = trainLinearReg(X, y, lambda);


%%  Plot fit over the data
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
hold on;
plot(X, [ones(m, 1) X]*theta, '--', 'LineWidth', 2)
hold off;


%% Learning Curve

lambda = 0.10;
[error_train, error_val] = ...
    learningCurve([ones(m, 1) X], y, ...
                  [ones(size(Xval, 1), 1) Xval], yval, ...
                  lambda);
              
%% Plot learning curve         

plot(1:m, error_train, 1:m, error_val);
title('Learning curve for linear regression')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 150])

%% Table of errors

fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1:m
    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
end