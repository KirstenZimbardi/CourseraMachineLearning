%% Initialization
clear ; close all; clc

%% data

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

%[m, n] = size(X);
%x = [ones(m,1) X];
x = X;
[m, n] = size(x);
itheta = zeros(n,1);
%xt = X';

%% submit function

%lrCostFunction([0.25 0.5 -0.5]', X, y, 0.1);

itheta = [0.25 0.5 -0.5]'
lambda = 0.1


%% g function
z = x * itheta;
h = sigmoid(z);

%% Cost

E = (-y .* log(h)) - ((1-y) .* log(1-h));

%J = 1/m * sum(E)

reg = sum(itheta(2:n) .* itheta(2:n));

J = 1/m * sum(E) + (lambda/(2*m) * reg)

%% gradient


%grad = 1/m * (X' * (h - y));
grad = zeros(size(itheta));
grad(2:n,:) = grad(2:n,:) + ((lambda/m) * itheta(2:n));

