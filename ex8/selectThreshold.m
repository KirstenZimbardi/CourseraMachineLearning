function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;

for epsilon = min(pval):stepsize:max(pval)

    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions


% cheat - reverse engineering from labels
%pos = find(yval);
%ppos = pval(pos);
%thresh = max(ppos) %NB probably false negatives given the distribution

    pred = (pval < epsilon);

    error = pred - yval;
    false_negative = find(error < 0);
    false_positive = find(error > 0);

    fn = length(false_negative);
    fp = length(false_positive); % instruction pdf has:
    %fp = sum((pred == 1) & (yval == 0))
    % so fn would similarily be:
    %fn = sum((pred == 0) & (yval == 1))
    % ie 2 lines replaces the above 5 lines using find and length
    % kinda handy to have the indecies for the fp and fn

    tp = length(find(yval & yval == pred));

    F1 = (2 * (tp/(tp+fp)) * (tp/(tp+fn)) ) / ( (tp/(tp+fp)) + (tp/(tp+fn)) );

    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
    
    % had to take break condition out to get submit successful
    %if F1 < bestF1 
    %    break
    %end
            
end

end
