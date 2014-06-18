% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Hyun Oh Song
% 
% This file is part of the Song-ICML2014 code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

conf = voc_config();

sharpness = '100';
loss_type = 'SmoothHinge';

svm_C = '0.001';
bias_mult = '10';
pos_loss_weight = '2';

class_list = conf.pascal.VOCopts.classes;
classes_to_evaluate = 1:20;
data = {};

mu = {'0.01'};
topK  = {'15'};

alpha = '0.95';
K1 = '0.5';
K2 = '1';
nms_threshold = '0.3';

for mu_i = 1:length(mu)
  for topK_i = 1:length(topK)
    
    ap_string = []; ap_only = [];
    
    for classid = classes_to_evaluate
      
      load_filename = [conf.paths.model_dir, ...
        class_list{classid}, '_pr_test_2007_',...
        'latentiter_*' ,... 
        '_clss_' class_list{classid}, ... 
        '_C_'  svm_C, ...
        '_B_'  bias_mult, ...
        '_w1_' pos_loss_weight, ... 
        '_losstype_' loss_type, ...
        '_sharpness_' sharpness, ...
        '_mu_' num2str(str2double(mu{mu_i})), ...
        '_alpha_' num2str(str2double(alpha)),...
        '_K1_' num2str(str2double(K1)), ...
        '_K2_' num2str(str2double(K2)), ...
        '_nms_' num2str(nms_threshold), ...
        '_topK_' num2str(str2double(topK{topK_i})),...
        '_20x1_smooth_topK_bagmine_greedycover_final'];

      iterations = [];
      d = dir(load_filename);
      for i = 1:length(d)
        this_name = d(i).name;
        bars = strfind(this_name, '_');
        iteration_id = str2double(this_name(bars(5)+1 : bars(6)-1));
        iterations = [iterations; iteration_id];
      end
      
      if isempty(d)
        error('not computed');
        ap = nan;
      else
        highest_iteration_fileid = find(iterations == max(iterations));
        load_filename = d(highest_iteration_fileid).name;

        if mu_i == 3 && topK_i == 3
          disp(load_filename);
        end

        load([conf.paths.model_dir, load_filename], '-mat');
      end
      
      ap_string = [ap_string, ' ', class_list{classid}, ' ', num2str(ap)];
      ap_only = [ap_only, ap];    
    end
      
    data{end+1} = [ ' C '  svm_C, ...
          ' B '  bias_mult, ...
          ' w1 ' pos_loss_weight, ... 
          ' losstype ' loss_type, ...
          ' sharpness ' sharpness, ...
          ' mu ' num2str(str2double(mu{mu_i})), ...             
          ' alpha ' num2str(str2double(alpha)),...
          ' K1 ' num2str(str2double(K1)), ...
          ' K2 ' num2str(str2double(K2)), ...
          ' nms ' num2str(nms_threshold), ...                          
          ' topK ' num2str(str2double(topK{topK_i})),...                
          ' AP '  ap_string,...
          ' mAP ' num2str(mean(ap_only))];
  end
end


data = data';
filename = 'smooth_lsvm_topK_bagmine_greedycover_maxiter_june17.dat';
fid = fopen(filename, 'w');
for row = 1:length(data)
  fprintf(fid, '%s\n', data{row});
end
fclose(fid);