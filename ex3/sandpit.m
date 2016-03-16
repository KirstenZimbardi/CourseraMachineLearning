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

[m, n] = size(X);
%X = [ones(m,1) X]
%[m, n] = size(X);
%submission code
%lrCostFunction([0.25 0.5 -0.5]', X, y, 0.1);
theta = [0.25 0.5 -0.5]'
lambda = 0.1
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

%% my function code

% g
z = X * theta;
h = sigmoid(z);

%% Cost

E = (-y .* log(h)) - ((1-y) .* log(1-h));

%J = 1/m * sum(E)

reg = sum(theta(2:n) .* theta(2:n));

J = 1/m * sum(E) + (lambda/(2*m) * reg)

%% gradient
Es = h-y
Egrad = repmat(Es,1,n)
%Egrad(:,1) = 0

Error = Egrad .* X

%% 
%grad = 1/m * (X' * (h - y));

grad = zeros(size(theta))

%grad = (1/m * sum(Error)) + ((lambda/m) * theta)

error = (1/m * sum(Error))'
temp = theta; 
temp(1) = 0;   
reg = ((lambda/m) * temp)

grad = error + reg


%% 

temp = theta; 
temp(1) = 0;   

grad = 1/m * (Error + ((lambda/m) * temp))

grad = grad + ((lambda/m) * temp)


%% 

for j = 1:n
    Error(j,:) = Egrad .* X(:,j);
    grad(j,:) = 1/m * sum(Error(j,:));
end
%% 





