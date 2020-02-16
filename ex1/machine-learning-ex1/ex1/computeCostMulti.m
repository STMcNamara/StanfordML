function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% Compute the hypothesis vector, h
h = X * theta;

% Compute the square error
error = h - y;
error_sqr = error.^2;

% Sum the vector and scale to calculate cost, J
error_sqr_sum = sum(error_sqr);
J = (1/(2*m))*error_sqr_sum;
end
