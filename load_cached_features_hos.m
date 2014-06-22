function d = load_cached_features_hos(dataset, year, ids, opts)

if ~exist('opts', 'var') || isempty(opts)
  opts.layer       = 'fc7';
  opts.fine_tuned  = 0;
  opts.use_flipped = 0;
end

% feat_cache/voc_[year]_flip_[flip]_layer_[layer]_finetuned_[finetuned]/[dataset]

base_path = sprintf('feat_cache/voc_%s_flip_%d_layer_%s_finetuned_%d', ...
                    year, opts.use_flipped, opts.layer, opts.fine_tuned);

file = sprintf('%s/%s/%s', base_path, dataset, ids);

if exist([file '.mat'], 'file')
  d = load(file);
  
  % fc7 features are already dealt with this.
  if ~strcmp(opts.layer, 'fc7')
    % standardize boxes to double (for overlap calculations, etc.)
    d.boxes = double(d.boxes);

    % features are not rectified
    d.feat = max(0, d.feat);
  end
  
else
  % if fc7 features doesn't exist, load pool5 and convert into fc7
  p5_base_path = sprintf('feat_cache/voc_%s_flip_%d_layer_%s_finetuned_%d', ...
                    year, opts.use_flipped, 'pool5', opts.fine_tuned);
 
  file = sprintf('%s/%s/%s', p5_base_path, dataset, ids);
  d = load(file);
  
  % standardize boxes to double (for overlap calculations, etc.)
  d.boxes = double(d.boxes);

  % features are not rectified
  d.feat = max(0, d.feat);
  
  %fprintf('loading fc6 layer params\n');
  fc6_layer = load([p5_base_path '/net_weights/weights_fc6.mat']);
  fc6_layer.weights = squeeze(fc6_layer.weights);
  d.feat = max(0, bsxfun(@plus, d.feat*fc6_layer.weights, fc6_layer.biases));
  
  %fprintf('loading fc7 layer params\n');
  fc7_layer = load([p5_base_path '/net_weights/weights_fc7.mat']);
  fc7_layer.weights = squeeze(fc7_layer.weights);
  d.feat = max(0, bsxfun(@plus, d.feat*fc7_layer.weights, fc7_layer.biases));
  
  file = sprintf('%s/%s/%s', base_path, dataset, ids);
  save([file '.mat'], '-struct', 'd');
end
