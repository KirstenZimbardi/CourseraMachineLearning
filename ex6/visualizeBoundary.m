function visualizeBoundary(X, y, model, varargin)
%VISUALIZEBOUNDARY plots a non-linear decision boundary learned by the SVM
%   VISUALIZEBOUNDARYLINEAR(X, y, model) plots a non-linear decision 
%   boundary learned by the SVM and overlays the data on it

%Note: Error in visualizeBoundary.m. Change the call to contour() like this:
%contour(X1, X2, vals, [1 1], 'b');
%(This change removes the attribute 'Color', and changes the contour interval. 
%Note that [0.5 0.5] also works and is more logical, since "vals" has range [0..1])
%This issue can cause either the "hggroup" error message, 
%or the decision boundaries to not be displayed, 
%or possibly cause Octave 3.8.x to crash when running ex6.m.

% Plot the training data on top of the boundary
plotData(X, y)

% Make classification predictions over a grid of values
x1plot = linspace(min(X(:,1)), max(X(:,1)), 100)';
x2plot = linspace(min(X(:,2)), max(X(:,2)), 100)';
[X1, X2] = meshgrid(x1plot, x2plot);
vals = zeros(size(X1));
for i = 1:size(X1, 2)
   this_X = [X1(:, i), X2(:, i)];
   vals(:, i) = svmPredict(model, this_X);
end

% Plot the SVM boundary
hold on
contour(X1, X2, vals, [0 0], 'Color', 'b');
hold off;

end
