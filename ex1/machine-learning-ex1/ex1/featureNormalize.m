function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% Useful values
m = size(X,1)

% Compute mean and standard deviation
mu = mean(X);
sigma = std(X);

% Computer mu and sigma vectors
mu_matrix = ones(m, 1) * mu;  
sigma_matrix = ones(m, 1) * sigma;

% Normalise by subtracting mu and elementwise division of sigma
X_norm = (X - mu_matrix) ./ sigma_matrix;


end
