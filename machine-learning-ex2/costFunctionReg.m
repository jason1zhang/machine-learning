function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y);      % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% z = theta' * X';            % theta (n x 1), X (m x n), so z (1 x m)
z = X * theta;

% J is a scalar value
% J = ((log(sigmoid(z)) * y * (-1)) - (log(1 - sigmoid(z)) * (1 - y))) ./ m;
J = sum(((log(sigmoid(z)) .* y * (-1)) - (log(1 - sigmoid(z)) .* (1 - y)))) / m;

n = length(theta);                  % number of features
reg_theta = theta(2:n,:);           % not regularize theta(1)
reg_J_term = reg_theta' * reg_theta * lambda / (2 * m);

J = J + reg_J_term;

% z = z';                                     % z' (m x 1)
grad = (X' * (sigmoid(z) - y)) / m;         % grade (n x 1)
temp = grad(1);                             % tempararily store grad(1) for not being overwrite by the next step using vectorization
reg_grade_term = theta * lambda / m;
grad = grad + reg_grade_term;
grad(1) = temp;


% =============================================================

end
