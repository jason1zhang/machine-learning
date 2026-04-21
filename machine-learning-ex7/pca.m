function [U, S] = pca(X)
%PCA Run principal component analysis on the dataset X
%   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%

% Useful values
[m, n] = size(X);

% You need to return the following variables correctly.
U = zeros(n);
S = zeros(n);

% ====================== YOUR CODE HERE ======================
% Instructions: You should first compute the covariance matrix. Then, you
%               should use the "svd" function to compute the eigenvectors
%               and eigenvalues of the covariance matrix. 
%
% Note: When computing the covariance matrix, remember to divide by m (the
%       number of examples).
%

%{
fprintf('\nSize of X = (%d, %d)\n', size(X,1), size(X,2));
X(1,:)
%}

% Compute the covariance matrix
sigma = (X' * X) ./ m;

% Compute the eigenvectors and eigenvalues of the covariance matrix sigma
[U, S, ~] = svd(sigma);

%{
fprintf('\nSize of U = (%d, %d)\n', size(U,1), size(U,2));
U
fprintf('\nSize of S = (%d, %d)\n', size(S,1), size(S,2));
S
%}

% =========================================================================

end
