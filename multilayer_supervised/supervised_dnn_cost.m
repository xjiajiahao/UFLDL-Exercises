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
    zCurLayer = stack{ii}.W * hAct{ii} + repmat(stack{ii}.b, 1, m);
    % sigmoid function
    hAct{ii + 1} = 1 ./ (1 + exp(-zCurLayer));
    % compute weightDecay
end
% the probabilities (output_dim * m)
zCurLayer = stack{numHidden + 1}.W * hAct{numHidden + 1}+ repmat(stack{numHidden + 1}.b, 1, m);

pred_prob = 1 ./ (1 + exp(-zCurLayer));

%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%
cost = 0;
[~,pred] = max(pred_prob);
squaredError = (pred - labels).^2;

weightDecay = 0;
for ii = 1 : numHidden + 1
    squaredWeight = (stack{ii}.W).^2;
    weightDecay = weightDecay + sum(weightDecay(:));
end
cost = cost + 1 / (m * 2) * sum(squaredError(:)) + ei.lambda / 2 * weightDecay;

%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%

for ii = numHidden + 1 : -1 : 1
    curSize = size(stack{ii}, 1);
    % initialize deltaLplus1
    if ii == numHidden + 1
        devirate = (pred_prob) .* (1 - pred_prob);
        deltaLplus1 = - (repmat(labels, curSize, 1) - pred_prob) .* devirate;
        gradStack{ii}.W = deltaLplus1 * pred_prob';
    else
        gradStack{ii}.W = deltaLplus1 * (hAct{ii + 1})';
    end
    devirate = (hAct{ii}) .* (1 - hAct{ii});
    gradStack{ii}.b = deltaLplus1;
    % update deltaLplus1
    deltaL = ((stack{ii}.W)' * deltaLplus1) .* devirate;
    deltaLplus1 = deltaL;
end
%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end
