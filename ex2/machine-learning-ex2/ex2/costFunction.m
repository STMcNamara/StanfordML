function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% Compute the z vector:
z = X * theta;

% Compute the hypothese vector h, by applying the sigmoid function
h = sigmoid(z);

% Compute J using vectorized implementation
J = (1/m) * (-y' * log(h) - (1-y)' * log(1-h));

% Compute the dradient using a vectorized implementation
grad = ((1/m)*X')*(sigmoid(X*theta) - y);


end
