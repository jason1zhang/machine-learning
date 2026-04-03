function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
prediction = X * theta;                     % Compute the prediction (m x 1 vector) using vectorization
squared_error = (prediction - y) .^ 2;      % Compute the squared_error (m x 1 vector)
% J = (1/2*m) * (ones(m,1)' * squared_error);   % Compute the cost for linear regression
J = (1/(2*m)) * sum(squared_error);   % Compute the cost (scalar real number) for linear regression

% =========================================================================

end
