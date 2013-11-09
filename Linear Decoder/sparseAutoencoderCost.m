
function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, rho, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 

%feedforward 
[m n] = size(data(:, :));
% for i = 1 : n
%     a1 = data(:, i);
%     A1(:, i) = a1;
%     z2 = W1 * a1 + b1;
%     a2 = sigmoid(z2);
%     A2(:, i) = a2;
%     z3 = W2 * a2 + b2;
%     a3 = sigmoid(z3);
%     A3(:, i) = a3;
%     J(i) = sum((a3 - a1) .^ 2) / 2; %least-square cost function
% end;
a1 = data;
% A1 = a1;
z2 = W1 * a1 + repmat(b1, 1, n);
a2 = sigmoid(z2);
% A2 = a2;
z3 = W2 * a2 + repmat(b2, 1, n);
a3 = sigmoid(z3);
% A3 = a3;
J = sum((a3 - a1) .^ 2) / 2;

rho_average = mean(a2, 2);
mean_square_error = mean(J);
weight_decay = (sum(sum(W1 .^ 2)) + sum(sum(W2 .^ 2))) * lambda / 2 ;
KL_divergence = rho * log(rho ./ rho_average) + (1 - rho) * log((1 - rho) ./ (1 - rho_average));
sparsity_constraint = beta * sum(KL_divergence);
cost = mean_square_error + weight_decay + sparsity_constraint;

%backpropagation

% for i = 1 : n
%     a1 = A1(:, i); % activations of the layer 1 WRT the ith train data
%     a2 = A2(:, i);
%     a3 = A3(:, i);
%     
%     delta3 = -(a1 - a3) .* a3 .* (1 - a3); % the responsibilite to error of output-layer
%     
%     sparity_term = beta * (-rho ./ rho_average + (1 - rho) ./ (1 - rho_average));
%     
%     delta2 = (W2' * delta3 + sparity_term) .* a2 .* (1 - a2); % the responsibilite to error of hidden-layer
%     
%     J_WRT_W1 = delta2 * a1'; %derivation of J with respect to W1
%     J_WRT_b1 = delta2;
%     J_WRT_W2 = delta3 * a2';
%     J_WRT_b2 = delta3;
%     W1grad = W1grad + J_WRT_W1;
%     W2grad = W2grad + J_WRT_W2;
%     b1grad = b1grad + J_WRT_b1;
%     b2grad = b2grad + J_WRT_b2;
% end;

delta3 = -(a1 - a3) .* a3 .* (1 - a3);
sparity_term = beta * (-rho ./ rho_average + (1 - rho) ./ (1 - rho_average));
delta2 = (W2' * delta3 + repmat(sparity_term, 1, n)) .* a2 .* (1 - a2);
J_WRT_W1 = delta2 * a1'; 
J_WRT_b1 = delta2;
J_WRT_W2 = delta3 * a2';
J_WRT_b2 = delta3;
W1grad = W1grad + J_WRT_W1;
W2grad = W2grad + J_WRT_W2;
% b1grad = b1grad + J_WRT_b1;
% b2grad = b2grad + J_WRT_b2;

W1grad = W1grad / n + lambda * W1;
W2grad = W2grad / n + lambda * W2;
b1grad = sum(J_WRT_b1, 2) / n;
b2grad = sum(J_WRT_b2, 2) / n;



%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end
