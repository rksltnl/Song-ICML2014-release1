function eval_classes_20x1_smooth_lsvm_topK_bagmine_greedycover_new(...
         classid, svm_mu, topK, alpha, K1, K2, nms_threshold)
                  
% HOS: find the highest latent_iter and only evaluate mAP on it.       
%       the ideas is since I only want the maximum mAP, its's quicker to
%         go back and evaluate mAP on it's precursors.
       
if ischar(classid),     classid = str2double(classid); end
if ischar(svm_mu),      svm_mu  = str2double(svm_mu);  end
if ischar(topK),        topK = str2double(topK); end
if ischar(alpha),       alpha = str2double(alpha); end
if ischar(K1),          K1 = str2double(K1); end
if ischar(K2),          K2 = str2double(K2); end
if ischar(nms_threshold), nms_threshold = str2double(nms_threshold); end

conf = voc_config();

sharpness = '100';
loss_type = 'SmoothHinge';

svm_C = '0.001';
bias_mult = '10';
pos_loss_weight = '2';

class_list = conf.pascal.VOCopts.classes;

load_filename = ['latentiter_*' ,...
  '_clss_' class_list{classid}, ... 
  '_C_'  svm_C, ...
  '_B_'  bias_mult, ...
  '_w1_' pos_loss_weight, ... 
  '_losstype_' loss_type, ...
  '_sharpness_' sharpness, ...
  '_mu_' num2str(svm_mu), ...
  '_alpha_' num2str(alpha),...
  '_K1_' num2str(K1), ...
  '_K2_' num2str(K2), ...
  '_nms_' num2str(nms_threshold), ...
  '_topK_' num2str(topK),...
  '_20x1_smooth_topK_bagmine_greedycover_final.mat'];

iterations = [];
d = dir(['repro/2007/', load_filename]);
for i = 1:length(d)
  this_name = d(i).name;
  bars = strfind(this_name, '_');
  iteration_id = str2double(this_name(bars(1)+1 : bars(2)-1));
  iterations = [iterations; iteration_id];
end
  
highest_iteration_fileid = find(iterations == max(iterations));
load_filename = d(highest_iteration_fileid).name;

% strip .mat from load_filename
[~,append_string] = fileparts(load_filename); 
pr_filename = [class_list{classid}, '_pr_test_2007_', append_string];

if exist([conf.paths.model_dir, pr_filename], 'file')~=0
  fprintf('already evaluated. done\n');
else
  load([conf.paths.model_dir, load_filename], 'models');
  res = test_classes_hos_append_filename(models, 'test', '2007', append_string);
  fprintf('20x1 Smooth LSVM class: %s, ap: %f\n', models{1}.class, res.ap);
end