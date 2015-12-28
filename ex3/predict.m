function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

X = [ones(m, 1) X];

% confidences = sigmoid(Theta2 * sigmoid(Theta1 * X))

% Theta1 in R_25x401
% Theta2 in R_10x26
% X in R_5000x401

a_1 = X;
z_2 = Theta1 * a_1'; % z_2 in R_25x5000
a_2 = sigmoid(z_2)'; % a_2 in R_5000x25
a_2 = [ones(size(a_2, 1), 1) a_2]; % a_2 in R_5000x26
z_3 = Theta2 * a_2'; % z_2 in R_10x5000
a_3 = sigmoid(z_3);

confidences = a_3';

for i = 1:m,
  [v iv] = max(confidences(i, :));
  p(i) = iv;
end


% =========================================================================


end
