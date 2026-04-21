function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Part 1: Compute the cost of regularized linear regression
%{
fprintf('\n size of X = (%d %d) \n',size(X,1), size(X,2));
X
fprintf('\n size of y = (%d %d) \n',size(y,1), size(y,2));
y
fprintf('\n size of theta = (%d %d) \n',size(theta,1), size(theta,2));
theta
%}

prediction = X * theta;                         % Compute the prediction using vectorization
squared_error = (prediction - y) .^ 2;          % Compute the squared_error
J = (ones(m,1)' * squared_error) / (2*m);       % Compute the cost for linear regression

n = length(theta);                              % number of features
reg_theta = theta(2:n,:);                       % not regularize theta(1)
reg_J_term = reg_theta' * reg_theta * lambda ./ (2 * m);

J = J + reg_J_term;

% Part 2: Compute the gradient of regularized linear regression
grad = (X' * (X * theta - y)) ./ m;         % grad (n x 1) vector
grad_1 = grad(1);                           % tempararily store grad(1) for not being overwrite by the next step using vectorization
reg_grad_term = theta * lambda ./ m;        % Compute the regularization term for gradient
grad = grad + reg_grad_term;
grad(1) = grad_1;


% =========================================================================

grad = grad(:);

end
