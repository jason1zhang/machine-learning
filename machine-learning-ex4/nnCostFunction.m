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

% fprintf('\n start debugging ...\n');
% fprintf('\n m = %d',m);
         
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
%
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
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Part 1: Compute the cost in the variable J
a1 = [ones(m,1), X];        % a1 (5000 * 401), aka (m x (n + 1))
z2 = Theta1 * a1';          % z2 (25 * 5000), aka (number of units in hidden layer x m)

a2 = sigmoid(z2);           % a2 (25 * 5000), aka (number of units in hidden layer x m)
a2 = [ones(m,1), a2'];      % a2 (5000 * 26), aka (m x (number of units in hidden layer + 1))
z3 = Theta2 * a2';          % z3 (10 * 5000), aka (number of labels x m)

a3 = sigmoid(z3)';          % a3 (5000 * 10), aka (m x number of labels)

% Recode vector y to the (m x num_labels) matrix Y
y = y';                     % Convert to (1 x m) vector

I = eye(num_labels);        % Create a (num_labels x num_labels) identity matrix
Y = zeros(num_labels, m);   % Initialize Y (num_labels x m) matrix
for i = 1:m
    Y(:,i) = I(:,y(i));
end
% fprintf('\n size of Y = %d',size(Y));

% Compute the un-regularized J value (J is the scalar value
J = sum(diag(((log(a3) * Y * (-1)) - (log(1 - a3)) * (1 - Y))) ./ m);
% fprintf('\n J = %d',J);

% Part 2: Implement regularization with the cost function
reg_theta1 = Theta1(:,2:size(Theta1,2));
reg_theta2 = Theta2(:,2:size(Theta2,2));

%{
fprintf('\n size of reg_theta1 = (%d %d) \n',size(reg_theta1,1), size(reg_theta1,2));
fprintf('\n size of reg_theta2 = (%d %d) \n',size(reg_theta2,1), size(reg_theta2,2));
%}

% Use function diag to the value of squares
reg_J_term1 = sum(diag(reg_theta1' * reg_theta1) * lambda ./ (2 * m));  
reg_J_term2 = sum(diag(reg_theta2' * reg_theta2) * lambda ./ (2 * m));
J = J + reg_J_term1 + reg_J_term2;

% Part 3: Implement the back-propagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. 
delta_3 = a3 - Y';      % delta_3 (5000 * 10), aka (m x number of labels)

%{
fprintf('\n size of delta_3 = (%d %d) \n',size(delta_3,1), size(delta_3,2));
fprintf('\n size of Theta2 = (%d %d) \n',size(Theta2,1), size(Theta2,2));
fprintf('\n size of z2 = (%d %d) \n',size(z2,1), size(z2,2));
%}

delta_2 = delta_3 * Theta2(:,2:end) .* sigmoidGradient(z2');    % delta_2 (5000 * 25), aka (m x (number of units in hidden layer))
% delta_2 = delta_2(:,2:end);                                   % No need to remove delta_2(1), such that delta_2 (5000 * 25), aka (m x (number of units in hidden layer))

% Accumulate the gradient
Theta1_grad = Theta1_grad + delta_2' * a1;       % Theta1_grad (25 * 401), aka ((number of units in hidden layer) * (n + 1))
Theta2_grad = Theta2_grad + delta_3' * a2;       % Theta2_grad (10 * 26), aka ((number of labels) x (number of units in hidden layer + 1))

% Obtain the (unregularized) gradient
Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;

% Part 4: Implement regularization with the gradients
% Regularize the Theta1_grade
Theta1_grad_reg = Theta1_grad(:,2:end);                 % Not to regularize the first column (bias term)
Theta1_grad_reg_term = Theta1(:,2:end) * (lambda / m);  % Regularized term for Theta1_grad
Theta1_grad_reg = Theta1_grad_reg + Theta1_grad_reg_term;
Theta1_grad = [Theta1_grad(:,1), Theta1_grad_reg];

% Regularize the Theta2_grade
Theta2_grad_reg = Theta2_grad(:,2:end);                 % Not to regularize the first column (bias term)
Theta2_grad_reg_term = Theta2(:,2:end) * (lambda / m);  % Regularized term for Theta2_grad
Theta2_grad_reg = Theta2_grad_reg + Theta2_grad_reg_term;
Theta2_grad = [Theta2_grad(:,1), Theta2_grad_reg];



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
