function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
%inputVars = 'X: %d';
%fprintf(inputVars, X);
disp(X); disp(y);
fprintf('Theta'); disp(theta); disp(alpha);disp(num_iters);
counter = 0;
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
%Theta is a 2x1 vector
theta0 = theta(1,:);  %These are going to be all ones
theta1= theta(2,:); % This is our X1 parameter

%printOutput1 = 'Theta0 is %d and Theta1 is %d\n';
%fprintf(printOutput1, theta0, theta1);
%fprintf(theta1);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Notes%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%Theta     X
%  |       |    (Theta is a 2x1 matrix and X is a 94x2)
% 2x1     94x2
%
%
% So we do X*Theta to receive a 94x1 vector
%   (94x2) * (2x1) => (94x1) matrix
%   
%   y is a 94x1 matrix and we need a 
%
%
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%End of Notes%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%while i < num_iters, %%% Commented out because of for loop above
    %   j = 1;
	theta0 = theta0 - (alpha * (1/m) * sum(X*theta-y));
    
    
    %%%%%%%%%%%%%%%%%%%%%Debugging theta1%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %   j = j + 1
    %fprintf('Sum of squares portion:\n');
    %disp(sum(X*theta-y));
    
    %fprintf('Derivative Term:\n');                 
    %disp(X(:,2));
    %fprintf('Derivative Term in summation:\n');
    %disp(sum(X(:,2)));
    
    
    %t1_sum_inc =  sum((X*theta-y).*X(:,2));
    
    %t1_sum_exc = sum((X*theta-y))*X(:,2);
    
    %%%%%%%%%%%%%%%%%%%%%Concluding Equation%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
	theta1 = theta1 - alpha * (1/m) * sum((X*theta-y).*X(:,2));
    % New  = old    - alpha * costFunction
    %                           |
    %                            -> 
	
    
    
	%theta(1,:) = theta0; %Initialized Above...
	%theta(2,:) = theta1;
    
    %disp(theta);
    
	counter = counter + 1;
	disp(theta0);
    disp(theta1);
    disp(counter);
    theta(1,:) = theta0;  %These are going to be all ones
    theta(2,:) = theta1; 




    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
