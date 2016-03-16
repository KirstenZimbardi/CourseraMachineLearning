1/(1+exp(-z))

%% 

E1 = E .* X(:,2);
E2 = E .* X(:,3);

% temp(1,1) = theta(1,1) - alpha * 1/m * sum(E);
% temp(1,2) = theta(1,2) - alpha * 1/m * sum(E1);
% temp(1,3) = theta(1,3) - alpha * 1/m * sum(E2);

% theta = temp;

temp(1,1) = 1/m * sum(E);
temp(1,2) = 1/m * sum(E1);
temp(1,3) = 1/m * sum(E2);
grad = temp;

% Eall = [E,E1,E2];


Egrad = h - y;
EgradX1 = Egrad .* X(:,2);
EgradX2 = Egrad .* X(:,3);
Egrad = EgradX1 + EgradX2;

temp(1,1) = 1/m * sum(E0);
temp(1,2) = 1/m * sum(E1);
temp(1,3) = 1/m * sum(E2);
grad = temp;


grad = 1/m * sum(Egrad);

%% 

E2(:,j) = Egrad .* X(:,j)

%% 
[m, n] = size(X);

%alpha = 0.05;
%theta = zeros(1,l(2));

% You need to return the following variables correctly 
theta = initial_theta;

z = X * theta;

h = sigmoid(z);

E = (-y .* log(h)) - ((1-y) .* log(1-h));

J = 1/m * sum(E);

%% 


grad = zeros(size(theta))

Egrad = h - y;

%% 


for j = 1:n
    Error(j,:) = Egrad .* X(:,j);
    grad(j,:) = 1/m * sum(Error(j,:));
end
    
grad

%% 

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);

%% 

prob = sigmoid([1 45 85] * theta);

prob = sigmoid(X * theta);

prediction = ge(prob,0.5);

%% 
[m, n] = size(X);
theta = initial_theta;

z = X * theta

h = sigmoid(z)

E = (-y .* log(h)) - ((1-y) .* log(1-h))
%% 

reg = sum(theta .* theta)

J = 1/m * sum(E) + (lambda/(2*m) * reg)
%% 

Egrad = h - y

%Error(1,1) = Egrad.* X(:,1)

for j = 1:n
    Error(j,:) = Egrad .* X(:,j);
    grad(j,:) = 1/m * sum(Error(j,:));
end
%% 

for j = 2:n
    grad(j,:) = grad(j,:) + ((lambda/m) * theta(j))
end


