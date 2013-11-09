function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer* (nearest to
% softmax Classifier)
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
[n m] = size(data);
groundTruth = full(sparse(labels, 1:m, 1));


%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%
a1 = data;
W1 = stack{1}.w;
b1 = stack{1}.b;
z2 = bsxfun(@plus, W1 * a1, b1);
a2 = sigmoid(z2);
W2 = stack{2}.w;
b2 = stack{2}.b;
z3 = bsxfun(@plus, W2 * a2, b2);
a3 = sigmoid(z3);
W3 = softmaxTheta;
b3 = 0;
z4 = bsxfun(@plus, W3 * a3, b3);
z4 = bsxfun(@minus, z4, max(z4, [], 1));
a4 = exp(z4);
a4 = bsxfun(@rdivide, a4, sum(a4, 1));

weight_decay = sum(sum(W1 .^ 2)) * lambda / 2 + sum(sum(W2 .^ 2)) * lambda / 2 + sum(sum(W3 .^ 2)) * lambda / 2;
cost = sum(sum(groundTruth .* log(a4))) / -m + weight_decay;

deltaTotal = -(groundTruth - a4) .^ 2 / 2; 
delta4 = -(groundTruth - a4); %partially derivation of deltaTotal on z4;
% delta3 = delta4 * W3 * (a3 .* (1 - a3)); % on z3
delta3 = W3' * delta4 .* (a3 .* (1 - a3)); % on z3
% delta2 = delta3 * W2 * (a2 .* (1 - a2)); % on z2
delta2 = W2' * delta3 .* (a2 .* (1 - a2)); % on z2

softmaxThetaGrad = delta4 * a3' / m + lambda * W3;
stackgrad{1}.w = delta2 * a1' / m + lambda * W1;
stackgrad{1}.b = mean(delta2, 2);
stackgrad{2}.w = delta3 * a2' / m + lambda * W2;
stackgrad{2}.b = mean(delta3, 2);

% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end