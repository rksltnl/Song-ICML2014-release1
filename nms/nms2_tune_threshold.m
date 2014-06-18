function [best_thresh_smaller, best_thresh_larger, best_ap, res] = nms2_tune_threshold(model, testset, year)

conf = voc_config('pascal.year', year, ...
                  'eval.test_set', testset);
cachedir = conf.paths.model_dir;                  
VOCopts  = conf.pascal.VOCopts;
load([cachedir model.class '_boxes_' testset '_' year]);

threshs = 0.2:0.05:0.4;
best_ap = 0;

res = zeros(0, 3);
for i = 1:length(threshs)
  thresh_smaller = threshs(i);
  for j = 1:length(threshs)
    thresh_larger = threshs(j);
    fprintf('trying thresholds (%.3f, %.3f)\n', thresh_smaller, thresh_larger);
    ap_auc = compute_at_nms_thresh(model, boxes, thresh_smaller, thresh_larger, VOCopts);
    res(end+1,:) = [thresh_smaller thresh_larger ap_auc];
    if ap_auc > best_ap
      best_thresh_smaller = thresh_smaller;
      best_thresh_larger = thresh_larger;
      best_ap = ap_auc;
      fprintf('!!! %s AP = %.3f @ threshs = (%.3f, %.3f)\n', model.class, best_ap, thresh_smaller, thresh_larger);
    end
  end
end



function ap_auc = compute_at_nms_thresh(model, boxes, thresh_smaller, thresh_larger, VOCopts)

image_ids = textread(sprintf(VOCopts.imgsetpath, VOCopts.testset), '%s');
% write out detections in PASCAL format and score
fid = fopen(sprintf(VOCopts.detrespath, 'comp3', model.class), 'w');
for i = 1:length(image_ids);
  bbox = boxes{i};
  keep = nms2(bbox, thresh_smaller, thresh_larger);
  bbox = bbox(keep,:);
  for j = 1:size(bbox,1)
    fprintf(fid, '%s %f %d %d %d %d\n', image_ids{i}, bbox(j,end), bbox(j,1:4));
  end
end
fclose(fid);

tic;
[recall, prec, ap_auc] = x10VOCevaldet(VOCopts, 'comp3', model.class, true);
%ap_auc = xVOCap(recall, prec);
