function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
num_labels = size(Theta2, 1);
total_layer=3;
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% Add ones to the X data matrix
a1 =[ones(m, 1) X];

%Feedforward the neural network to get activation h=a3
z2=Theta1*a1';
a2=sigmoid(z2)';
r=size(a2,1);
a2=[ones(r,1) a2];
z3=(Theta2*a2')';
a3=sigmoid(z3);

%Recode y into (m x num_labels) matrix with each row represents the output of binary vector
y_matrix=eye(num_labels)(y,:);

%Compute the cost
for i=1:num_labels
    J=J+1/m *(-(log(a3(:,i)))'* (y==i)-log(1-(a3(:,i)))'*(1-(y==i)));
end;

%Compute Theta1 square
theta1_sqr =Theta1(:,2:size(Theta1,2)) * Theta1(:,2:size(Theta1,2))';

%Compute Theta2 square
theta2_sqr =Theta2(:,2:size(Theta2,2)) * Theta2(:,2:size(Theta2,2))';

%compute sum of Theta1 square
sum_theta1=sum(sum(theta1_sqr .* eye(size(Theta1,1))));

%compute sum of Theta2 square
sum_theta2=sum(sum(theta2_sqr .* eye(size(Theta2,1))));

%Compute the regularization term separately
J=J+lambda/(2*m)*(sum_theta1+sum_theta2);


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

%Compute the error from output to hidden_layer
d_3=a3-y_matrix;

%Compute the error from hidden_layer to input_layer
d_2= d_3*Theta2(:,2:end) .* (sigmoid(z2') .* (1-sigmoid(z2')));

%This step calculates the product and sum of the errors
Delta_2 = d_3'*a2;
Delta_1 = d_2'*a1;

%Compute the gradients
Theta1_grad = (1/m) * Delta_1;
Theta2_grad = (1/m) * Delta_2;

%Compute the regularized term of gradients
regularizedTerm1 = lambda/m * Theta1(:,2:end);
regularizedTerm2 = lambda/m * Theta2(:,2:end);

%Combine gradients with regularized term
Theta1_grad =[Theta1_grad(:,1) (Theta1_grad(:,2:end) + regularizedTerm1)];
Theta2_grad =[Theta2_grad(:,1) (Theta2_grad(:,2:end) + regularizedTerm2)];

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
