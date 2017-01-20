function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
disp(theta);
%disp(X);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
% X Theta
% 90 x 2 2 x 1
%fprintf('Displaying X');
%disp(X)
z = (X*theta);



%X * theta
% 90 x 3 * 3 * 1

%fprintf('Displaying z to be put into hypothesis');
%disp(z);
h_theta_x = sigmoid(z);

%fprintf('H_Theta_X');
%disp(h_theta_x);


% 
% If prediction is one
%   h = -log(x)
% Otherwise
%   h = -log(1-x)

% X is a n x 3 matrix

%disp(y)

J= (1/m) .* sum(-y' * log(h_theta_x) - (1-y)' * log (1- h_theta_x)); 
disp(J)
disp(size(h_theta_x));
disp(size(y));
grad = (1/m).* (X' * (h_theta_x-y));


disp(size(grad));





% =============================================================

end
