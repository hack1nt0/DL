function [cost,grad,features] = sparseAutoencoderLinearCost(theta, visibleSize, hiddenSize, ...
                                                            lambda, rho, beta, data)
% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Copy sparseAutoencoderCost in sparseAutoencoderCost.m from your
%   earlier exercise onto this file, renaming the function to
%   sparseAutoencoderLinearCost, and changing the autoencoder to use a
%   linear decoder.
% -------------------- YOUR CODE HERE --------------------                                    

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

%feedforward 

[n m] = size(data(:, :));
a1 = data;
z2 = bsxfun(@plus, W1 * a1, b1);
a2 = sigmoid(z2);
z3 = bsxfun(@plus, W2 * a2, b2);
a3 = z3;
J = sum((a3 - a1) .^ 2) / 2;

rho_average = mean(a2, 2);
mean_square_error = mean(J);
weight_decay = (sum(sum(W1 .^ 2)) + sum(sum(W2 .^ 2))) * lambda / 2 ;
KL_divergence = rho * log(rho ./ rho_average) + (1 - rho) * log((1 - rho) ./ (1 - rho_average));
sparsity_constraint = beta * sum(KL_divergence);
cost = mean_square_error + weight_decay + sparsity_constraint;

%backpropagation

delta3 = -(a1 - a3);
sparity_term = beta * (-rho ./ rho_average + (1 - rho) ./ (1 - rho_average));
delta2 = (W2' * delta3 + repmat(sparity_term, 1, m)) .* a2 .* (1 - a2);
W1grad = delta2 * a1'; 
b1grad = delta2;
W2grad = delta3 * a2';
b2grad = delta3;
W1grad = W1grad / m + lambda * W1;
W2grad = W2grad / m + lambda * W2;
b1grad = mean(b1grad, 2);
b2grad = mean(b2grad, 2);

%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];
end

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end
