function nms_feat_debug(model, split, year, boxes)

conf = voc_config('pascal.year', year);

% ------------------------------------------------------------------------
% Set up training dataset
ids = textread(sprintf(conf.pascal.VOCopts.imgsetpath, split), '%s');
% ------------------------------------------------------------------------

feat_opts = model.opts;

for i = 1:length(boxes)
  if isempty(boxes{i}) || boxes{i}(1,end) < -0.5
    continue;
  end

  d = load_cached_features(split, year, ids{i}, feat_opts);

  keep = nms(boxes{i}, 0.3);
  nms_boxes = boxes{i}(keep, :);
  [~, loc] = ismember(nms_boxes(:,1:4), d.boxes, 'rows');
  
  F = d.feat(loc,:);
  F = bsxfun(@times, F, 1./sqrt(sum(F.^2, 2)));
  % normalized correlation matrix
  D = F*F';



  im = imread(sprintf(conf.pascal.VOCopts.imgpath, ids{i})); 
  for j = 1:size(nms_boxes, 1)-1
    if nms_boxes(j,end) < -0.5
      continue;
    end

    fprintf('%d/%d\n', j, size(nms_boxes, 1));
    bbox = double(nms_boxes(j,:));
    showboxesc(im, bbox, 'b', '-');
    inds = j+1:size(D,2);
    dists = D(j, inds);
    [dvals, ord] = sort(dists, 'descend');
    for k = 1:length(ord)
      ind = inds(ord(k));
      bbox2 = double(nms_boxes(ind,:));
      showboxesc([], bbox2, 'r', '-');
      text(bbox2(1), bbox2(2), sprintf('%.2f', dvals(k)), 'BackgroundColor', 'r');
    end
    pause;
  end
end
