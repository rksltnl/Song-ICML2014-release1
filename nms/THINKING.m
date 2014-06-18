% load all boxes
% for each image, set scores for non-max class to -inf
%
%
%
%

% Feature space NMS
%
% Debugging experiment
%  - choose a sample image from test 2007
%  - go through detections sorted from highest to lowest scoring
%  - for each high scoring detection, show the closest in feature space
%
%
%
%
%
% Run test on train+val set to get boxes from there
%
%
%
%
%
%


% Method 1:
%  two thresholds, one for smaller objects, one for bigger objects
%
% Method 2:
%  - given a TP window wi with score si compute a feature vector fij relative to wi for windows
%    wj for all j such that:
%      (sj < si) and (wj is a FP) and (wj \intersect wi is not empty)
%  - label wi as a positive example and all wj as negative examples
%  - train an SVM to classify these based on the features
%  - use a hard threshold on the SVM score, selected by grid search, for NMS
%
% Features
%  - ratio of scores: either si/sj  or  log(si/sj)
%    hopefully the ratio is robust to overfitting
%  - IoU between wi and wj: area(wi \cap wj) / area(wi \cup wj)
%  - fraction of wj inside wi: area(wi \cap wj) / area(wj)
%  - fraction of wi inside wj: area(wi \cap wj) / area(wi)
%  - ratio of areas: area(wi) / area(wj)
%  - logs of the previous ratios
%  - vector from center of wi to center of wj
%    - quantize angle 
%    - magnitude normalized by wi's diagonal
%
% Need a validation set for training and threshold selection
%  - check how much we're overfitting on train -- maybe it's ok to reuse it?
%  - 
