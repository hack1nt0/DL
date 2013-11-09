function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

% numCases = size(data, 2);
[n m] = size(data);
% groundTruth = sparse(labels, 1:numCases, 1);
groundTruth = full(sparse(labels, 1 : m, 1));
cost = 0;

thetagrad = zeros(size(theta));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.
prob = theta * data;
prob = bsxfun(@minus, prob, max(prob, [], 1)); % avoid exp overflow
prob = exp(prob);
prob = bsxfun(@rdivide, prob, sum(prob, 1)); % prob(i, j): data j's prob in class i;
weight_decay = sum(sum(theta .^ 2)) * lambda / 2;
cost = sum(sum(groundTruth .* log(prob))) / -m + weight_decay;

thetagrad = data * (groundTruth' - prob') ./ -m;
thetagrad = thetagrad + lambda * theta';%involve the weight decay 
thetagrad = thetagrad';
% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

