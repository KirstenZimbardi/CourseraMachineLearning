[m, n] = size(X);
itheta = zeros(n,1);
%xt = X';
z = X * itheta;
%% 

h = sigmoid(z);

%% Cost

E = (-y .* log(h)) - ((1-y) .* log(1-h))

J = 1/m * sum(E)
%% gradient


grad = 1/m * (X' * (h - y))

