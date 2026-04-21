function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

% z = theta' * X';            % theta (n x 1), X (m x n), so z (1 x m)
z = X * theta;            % theta (n x 1), X (m x n), so z (m x 1)

% J is a scalar value
J = sum(((log(sigmoid(z)) .* y * (-1)) - (log(1 - sigmoid(z)) .* (1 - y)))) / m;

n = length(theta);          % number of features
reg_theta = theta(2:n,:);   % not regularize the bias term theta(1)
reg_J_term = reg_theta' * reg_theta * lambda / (2 * m);

J = J + reg_J_term;

% z = z';                                     % z' (m x 1)
grad = (X' * (sigmoid(z) - y)) / m;         % grade (n x 1)
temp = grad(1);                             % tempararily store grad(1) for not being overwrite by the next step using vectorization
reg_grade_term = theta * lambda ./ m;
grad = grad + reg_grade_term;
grad(1) = temp;

% =============================================================

% grad = grad(:);

end
