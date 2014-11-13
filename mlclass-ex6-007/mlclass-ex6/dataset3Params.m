function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%initialize the step width and initial value
C_vect = [0.1; 0.3; 1; 3; 10];
sigma_vect = [0.1; 0.3; 1; 3; 10];
opt_c=0;
opt_sigma=0;
min_err=10000000;

%Mutate through 25 combination of C and sigma value with 5 steps each.
for i=1:5
    for j=1:5
          predictions= svmPredict(svmTrain(X,y,C_vect(i), @(x1,x2) gaussianKernel(x1,x2,sigma_vect(j))),Xval);
          err= mean(double(predictions ~= yval));
          
          %Compare err with min_err, and set min_err=err if err smaller then min_err
          if (err < min_err)
                min_err=err;
                opt_c = C_vect(i);
                opt_sigma=sigma_vect(j);
          end
    end
end

C=opt_c;
sigma=opt_sigma;


% =========================================================================

end
