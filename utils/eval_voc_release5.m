function res = eval_voc_release5()

testset = 'test';
year = '2007';
comp_id = 'comp3';

conf = voc_config('pascal.year', year, ...
                  'eval.test_set', testset);
cachedir = conf.paths.model_dir;                  
VOCopts  = conf.pascal.VOCopts;
image_ids = textread(sprintf(VOCopts.imgsetpath, testset), '%s');

rel5_box_file = '/work4/soursop-archive/data/rbg/rel5-dev/rc2/2007/%s_boxes_test_bboxpred_2007.mat';

for clss_ind = 1:length(VOCopts.classes)
  cls = VOCopts.classes{clss_ind};
  res_id = comp_id;
  res_fn = sprintf(VOCopts.detrespath, res_id, cls);

  load(sprintf(rel5_box_file, cls));

  % write out detections in PASCAL format and score
  fid = fopen(res_fn, 'w');
  for i = 1:length(image_ids)
    bbox = ds{i};
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
    [recall, prec, ap] = VOCevaldet(VOCopts, res_id, cls, true);
    ap_auc = xVOCap(recall, prec);

    % force plot limits
    ylim([0 1]);
    xlim([0 1]);

    print(gcf, '-djpeg', '-r0', [cachedir cls '_pr_' testset '_' year '.jpg']);
  end
  fprintf('!!! %s : %.4f %.4f\n', cls, ap, ap_auc);

  res(clss_ind).recall = recall;
  res(clss_ind).prec = prec;
  res(clss_ind).ap = ap;
  res(clss_ind).ap_auc = ap_auc;
end
