A = magic(10)
x = [1:10]'
% x = x'
%% 

v = zeros(10,1);
for i = 1:10
    for j = 1:10
        v(i) = v(i) + A(i,j) * x(j);
    end
end
%% 

v2 = A*x

%% 

v3 = Ax

%% 

v4 = x' * A

%% 

v5 = sum (A * x)
%% 
%% 

v = [1:7]'
w = [10:16]'

%% 

z = 0
for i = 1:7
    z = z + v(i) * w(i)
end

%% 

z1 = sum(v.*w)

%% 

z2 = w' * v

%% 

z3 = v * w'

%% 

z4 = w * v'


%% 
%% 

x = magic(7);

for i = 1:7
    for j = 1:7
        A(i,j) = log(x(i,j));
        B(i,j) = x(i,j)^2;
        C(i,j) = x(i,j)+1;
        D(i,j) = x(i,j)/4;
    end
end

%% 

x + 1
x/4









