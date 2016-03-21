%% Initialization
clear ; close all; clc

%% training data

load('ex3data1.mat'); % training data stored in arrays X, y

%% adding x0 to X
% usually don't do this - gets added in during each function so only
% changes the local X and not the original global X

[m, n] = size(X);
X = [ones(m,1) X];


%% submission data

% Random Test Cases
  X = [ones(20,1) (exp(1) * sin(1:1:20))' (exp(0.5) * cos(1:1:20))'];
  y = sin(X(:,1) + X(:,2)) > 0;
  Xm = [ -1 -1 ; -1 -2 ; -2 -1 ; -2 -2 ; ...
          1 1 ;  1 2 ;  2 1 ; 2 2 ; ...
         -1 1 ;  -1 2 ;  -2 1 ; -2 2 ; ...
          1 -1 ; 1 -2 ;  -2 -1 ; -2 -2 ];
  ym = [ 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 ]';
  t1 = sin(reshape(1:2:24, 4, 3));
  t2 = cos(reshape(1:2:40, 4, 5));



%% useful stuff

[m, n] = size(X); 

input_layer_size  = 400;  % 20x20 Input Images of Digits
num_labels = 10;          % 10 labels, from 1 to 10   

lambda = 0.1;


% You need to return the following variables correctly 
%J = 0;
%grad = zeros(size(theta));

%% regularised cost function

% initialise
theta = zeros(n,1);

[J, grad] = lrCostFunction(theta, X, y, lambda)


%% one vs all function 

[all_theta] = oneVsAll(X, y, num_labels, lambda)


%% predict using oneVsAll function

% p = predictOneVsAll(all_theta, X)

%m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];
[m, n] = size(X); 

%% 

z = X * all_theta';
h = sigmoid(z);
%% 

[M, I] = max(h, [], 2);

%% 

accuracy = I == y
test = sum(I == y)/m

%% 

sum(accuracy)

