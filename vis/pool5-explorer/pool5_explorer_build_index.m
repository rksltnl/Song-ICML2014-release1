function pool5_explorer_build_index(split, year)

% index.split
% index.year
% index.images
% index.features{i}.images_index
% index.features{i}.boxes
% index.features{i}.scores

% load S and mu
load cachedir/convnet-selective-search/pool5_cov;
[V, D] = eig(S);
D = diag(D);
PCA = diag(1./sqrt(D+0.001))*V';
ZCA = V*diag(1./sqrt(D+0.001))*V';

% take a model
% select highest scoring features

TOP_K = 1000;

feat_opts.layer       = 'pool5';
feat_opts.fine_tuned  = 1;
feat_opts.use_flipped = 0;

conf = voc_config('pascal.year', year);
VOCopts = conf.pascal.VOCopts;

ids = textread(sprintf(VOCopts.imgsetpath, split), '%s');
%ids = ids(randperm(length(ids), ceil(length(ids)/2)));

% select features
sel_features = 1:(6*6*256);

index.split = split;
index.year = year;
index.feat_opts = feat_opts;
index.images = ids;
features = cell(length(sel_features), 1);

for i = 1:length(features)
  features{i}.image_inds = [];
  features{i}.scores = [];
  features{i}.boxes = zeros(0, 4);
end

for i = 1:length(ids)
  tic_toc_print('%d/%d', i, length(ids));
  th = tic();
  d = load_cached_features(split, year, ids{i}, feat_opts);

  feat = reshape(d.feat, [], 256);
  feat = (ZCA*bsxfun(@minus, feat, mu)')';
  feat = reshape(feat, [], 6*6*256);
  %feat = reshape(feat, [], 256);
  %S = feat'*feat;
  %keyboard

  parfor f = sel_features
    threshold = min(features{f}.scores);
    if isempty(threshold)
      threshold = -inf;
    end
    sel_0 = find(feat(:,f) > threshold);
    if isempty(sel_0)
      continue;
    end
    bs = [d.boxes(sel_0,:) feat(sel_0,f)];
    sel = faster_nms(bs, 0.1);

    sel = sel_0(sel);
    sz = length(sel);

    new_image_inds = i*ones(sz,1);
    new_scores = feat(sel,f);
    new_boxes = d.boxes(sel,:);

    features{f}.image_inds = cat(1, features{f}.image_inds, ...
                                    new_image_inds);
    features{f}.scores = cat(1, features{f}.scores, ...
                                new_scores);
    features{f}.boxes = cat(1, features{f}.boxes, ...
                               new_boxes);

    [~, ord] = sort(features{f}.scores, 'descend');
    if length(ord) > TOP_K 
      ord = ord(1:TOP_K);
    end
    features{f}.image_inds = features{f}.image_inds(ord);
    features{f}.scores = features{f}.scores(ord);
    features{f}.boxes = features{f}.boxes(ord, :);
  end
  fprintf(' %.3fs\n', toc(th));

  if mod(i, 50) == 0
    index.features = features;
    save_file = sprintf('%s/pool5_explorer_index_%s_%s_fine_tuned_%d_zca', ...
                        'cachedir/convnet-selective-search', split, year, ...
                        feat_opts.fine_tuned);
    save(save_file, 'index');
    fprintf('checkpoint %d\n', i);
  end
end

index.features = features;
save_file = sprintf('%s/pool5_explorer_index_%s_%s_fine_tuned_%d_zca', ...
                    'cachedir/convnet-selective-search', split, year, ...
                    feat_opts.fine_tuned);
save(save_file, 'index');
