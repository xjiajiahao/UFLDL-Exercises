function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(poolRow, poolCol, featureNum, imageNum)
%

numImages = size(convolvedFeatures, 4);
numFilters = size(convolvedFeatures, 3);
convolvedDim = size(convolvedFeatures, 1);

pooledFeatures = zeros(convolvedDim / poolDim, ...
        convolvedDim / poolDim, numFilters, numImages);

% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the
%   (convolvedDim/poolDim) x (convolvedDim/poolDim) x numFeatures x numImages
%   matrix pooledFeatures, such that
%   pooledFeatures(poolRow, poolCol, featureNum, imageNum) is the
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region.
%
%   Use mean pooling here.

%%% YOUR CODE HERE %%%
for imageNum = 1:numImages
    for filterNum = 1:numFilters
        % use conv2 followed by subsampling and averaging
        pooledImage = zeros(convolvedDim - poolDim + 1);
        W = ones(poolDim);
        tmpConvolvedFeatures = squeeze(convolvedFeatures(:, :, filterNum, imageNum));
        pooledImage += conv2(tmpConvolvedFeatures, W, 'valid');
        % subsampling
        tmpPooledFeatures = pooledImage(1:poolDim:convolvedDim, 1:poolDim:convolvedDim);
        % averaging
        tmpPooledFeatures = tmpPooledFeatures / poolDim^2;
        pooledFeatures(:, :, filterNum, imageNum) = tmpPooledFeatures;
    end
end

end
