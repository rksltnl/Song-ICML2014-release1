function res = test_classes_hos_append_filename(...
                              models, testset, year, append_string)
                            
% Append given unique string to the filename so that we can load the 
%   boxes and ap results later.

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Hyun Oh Song
% 
% This file is part of the Song-ICML2014 code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

use_res_salt = true;
rm_res = true;
comp_id = 'comp4';

conf = voc_config('pascal.year', year, ...
                  'eval.test_set', testset);
cachedir = conf.paths.model_dir;                  
VOCopts  = conf.pascal.VOCopts;
image_ids = textread(sprintf(VOCopts.imgsetpath, testset), '%s');

% assume they are all the same
feat_opts = models{1}.opts;

try
  assert(1==2);
%   aboxes = cell(length(models), 1);
%   for i = 1:length(models)
%     load([cachedir models{i}.class 'hos_boxes_' testset '_' year]);
%     aboxes{i} = boxes;
%   end
catch
  ws = cat(2, cellfun(@(x) x.w, models, 'UniformOutput', false));
  ws = cat(2, ws{:});
  bs = cat(2, cellfun(@(x) x.b, models, 'UniformOutput', false));
  bs = cat(2, bs{:});

  aboxes = cell(length(models), 1);
  for i = 1:length(models)
    aboxes{i} = cell(length(image_ids), 1);
  end

  max_per_image = 100;
  max_per_set = ceil(100000/2500)*length(image_ids);
  top_scores = cell(length(models), 1);
  thresh = -inf(length(models), 1);
  box_counts = zeros(length(models), 1);

  for i = 1:length(image_ids)
    fprintf('%s: test (%s) %d/%d\n', procid(), testset, i, length(image_ids));
    d = load_cached_features_hos(testset, year, image_ids{i}, feat_opts);
    %d.feat = d.feat.^(feat_opts.pwr_xform);
    d.feat = xform_feat_custom(d.feat, feat_opts);
    %d.feat = o2p(d.feat);
    zs = bsxfun(@plus, d.feat*ws, bs);

    for j = 1:length(models)
      z = zs(:,j);
      I = find(~d.gt & z > thresh(j));
      aboxes{j}{i} = cat(2, single(d.boxes(I,:)), z(I));
      [~, ord] = sort(z(I), 'descend');
      ord = ord(1:min(length(ord), max_per_image));
      aboxes{j}{i} = aboxes{j}{i}(ord, :);

      box_counts(j) = box_counts(j) + length(ord);
      top_scores{j} = cat(1, top_scores{j}, z(I(ord)));
      top_scores{j} = sort(top_scores{j}, 'descend');
      if box_counts(j) > max_per_set
        top_scores{j}(max_per_set+1:end) = [];
        thresh(j) = top_scores{j}(end);
      end

      if 0
        im = imread(sprintf(VOCopts.imgpath, image_ids{i})); 
        keep = nms(aboxes{j}{i}, 0.3);
        for k = 1:min(10, length(keep))
          showboxes(im, aboxes{j}{i}(keep(k),1:4));
          title(sprintf('score: %.3f\n', aboxes{j}{i}(keep(k),end)));
          pause;
        end
      end
    end
  end

  for i = 1:length(models)
    for j = 1:length(image_ids)
      I = find(aboxes{i}{j}(:,end) < thresh(i));
      aboxes{i}{j}(I,:) = [];
    end

    save_file = [cachedir models{i}.class 'hos_boxes_' testset ...
      '_' year '_' append_string];
    
    try
      boxes = aboxes{i};
      save(save_file, 'boxes');
      clear boxes;
    catch err
      disp(err);
      keyboard;
    end
  end
end

for model_ind = 1:length(models)
  salt = sprintf('%d-%d', randi(100000), mod(round(models{model_ind}.w(randperm(length(models{model_ind}.w),1))*10000), 10000));
  if use_res_salt
    res_id = [comp_id '-' salt '-' append_string];
  else
    res_id = comp_id;
  end
  res_fn = sprintf(VOCopts.detrespath, res_id, models{model_ind}.class);

  % write out detections in PASCAL format and score
  fid = fopen(res_fn, 'w');
  for i = 1:length(image_ids);
    bbox = aboxes{model_ind}{i};
    keep = nms(bbox, 0.3);
    bbox = bbox(keep,:);
    for j = 1:size(bbox,1)
      fprintf(fid, '%s %f %d %d %d %d\n', image_ids{i}, bbox(j,end), bbox(j,1:4));
    end
  end
  fclose(fid);

  recall = [];
  prec = [];
  ap = 0;
  ap_auc = 0;

  do_eval = (str2num(year) <= 2007) | ~strcmp(testset, 'test');
  if do_eval
    % Bug in VOCevaldet requires that tic has been called first
    tic;
    [recall, prec, ap] = VOCevaldet(VOCopts, res_id, models{model_ind}.class, true);
    ap_auc = xVOCap(recall, prec);

    % force plot limits
    ylim([0 1]);
    xlim([0 1]);

    print(gcf, '-djpeg', '-r0', [cachedir models{model_ind}.class '_pr_' testset '_' year '.jpg']);
  end
  fprintf('!!! %s : %.4f %.4f\n', models{model_ind}.class, ap, ap_auc);

  % save results
  save([cachedir models{model_ind}.class '_pr_' testset '_' year '_' ...
                        append_string], 'recall', 'prec', 'ap', 'ap_auc');

  res(model_ind).recall = recall;
  res(model_ind).prec = prec;
  res(model_ind).ap = ap;
  res(model_ind).ap_auc = ap_auc;
  if rm_res
    delete(res_fn);
  end
end
