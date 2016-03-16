function plotData(x, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Find Indices of Positive and Negative Examples


pos = find(y == 1); 
neg = find(y == 0); 

figure;  
hold on;

plot(x(pos, 1), x(pos, 2), 'k+','LineWidth', 2, ...
     'MarkerSize', 7);
plot(x(neg, 1), x(neg, 2), 'ko', 'MarkerFaceColor', 'y', ...
     'MarkerSize', 7);

hold off;
