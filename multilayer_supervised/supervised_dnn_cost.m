function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

m = size(data, 2);
labels = labels';
%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);
%% forward prop
%%% YOUR CODE HERE %%%
hAct{1} = data;
for ii = 1 : numHidden
    zCurLayer = bsxfun(@plus, stack{ii}.W * hAct{ii}, stack{ii}.b);
    % sigmoid function
    hAct{ii + 1} = 1 ./ (1 + exp(-zCurLayer));
end
% the probabilities, the same as Softmax regression (output_dim * m)
zCurLayer = bsxfun(@plus, stack{numHidden + 1}.W * hAct{numHidden + 1}, stack{numHidden + 1}.b);

pred_prob = exp(zCurLayer);
sumP = sum(pred_prob);
pred_prob = bsxfun(@rdivide, pred_prob, sumP);

%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%

%%% Damn it, I wrote the wrong cost function and the wrong delta{nl}.
%%% squaredError, which is a wrong one
%cost = 0;
%[~,pred] = max(pred_prob);
%squaredError = (pred - labels).^2;

%weightDecay = 0;
%for ii = 1 : numHidden + 1
%    squaredWeight = (stack{ii}.W).^2;
%    weightDecay = weightDecay + sum(weightDecay(:));
%end
%cost = cost + 1 / (m * 2) * sum(squaredError(:)) + ei.lambda / 2 * weightDecay;

%%% cross-entropy
classIndices = sub2ind(size(pred_prob), labels, 1 : size(pred_prob, 2));
tmpCeCost = log(pred_prob(classIndices));
ceCost = -sum(tmpCeCost(:));

%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%
map = zeros(size(pred_prob));
map(classIndices) = 1;

for ii = numHidden + 1 : -1 : 1
    if ii == numHidden + 1
        % delta{nl}
        delta = pred_prob - map;
    end
    gradStack{ii}.b = sum(delta, 2);
    gradStack{ii}.W = delta * (hAct{ii})';
    if ii == 1
        break;
    end
    devirate = (hAct{ii}) .* (1 - hAct{ii});
    delta = (stack{ii}.W)' * delta .* devirate;
end
%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%
wCost = 0;
for ii = 1 : numHidden + 1
    squaredWeight = (stack{ii}.W).^2;
    wCost = sum(squaredWeight(:)) * ei.lambda / 2;
end
cost = ceCost + wCost;

% gradients of weight decay
for ii = numHidden : -1 : 1
    gradStack{ii}.W = gradStack{ii}.W + ei.lambda * stack{ii}.W;
%% reshape gradients into vector
[grad] = stack2params(gradStack);
end
