function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% Compute the z vector:
z = X * theta;

% Compute the hypothese vector h, by applying the sigmoid function
h = sigmoid(z);

% Compute the unregularised cost
J_unreg = (1/m) * (-y' * log(h) - (1-y)' * log(1-h));

% Set theta(0) equal to 0 and square
theta_0 = theta;
theta_0(1) = 0;
theta_sq = theta_0' * theta_0;

% Compute the regularization term
J_reg = (lambda / (2 * m)) * theta_sq;

% Combine for the regularized cost
J = J_unreg + J_reg;

% Compute the gradient for theta 1 through j using a vectorized implementation
grad = ((1/m)*X')*(sigmoid(X*theta) - y) + (lambda / m) * theta_0;

% Ensure grad returned as column vector
grad = grad(:)

end
