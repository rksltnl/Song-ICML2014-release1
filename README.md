Song-ICML2014
=============

### Citing the paper

	@inproceedings{Song-ICML2014,
		title = "On learning to localize objects with minimum supervision",
		booktitle = "International Conference on Machine Learning (ICML)",
		year = "2014", 
		author = "Hyun Oh Song and Ross Girshick and Stefanie Jegelka and Julien Mairal and Zaid Harchaoui and Trevor Darrell",
	}

### License

This code is released under the Simplified BSD License (refer to the
LICENSE file for details).

### System Requirements
* Linux
* MATLAB
 
### Install instructions

1. Download and install R-CNN code (https://github.com/rbgirshick/rcnn)
2. Following the instructions in R-CNN code, precompute fc7 features on the dataset (i.e. PASCAL VOC 2007)
3. Unpack this code.
4. Start Matlab where the unpacked code is.
5. Mex compile the simplex projection cc code. `>> curdir=pwd; cd('projsplx'); mex projsplx_c_float.cc; cd(curdir);`

### Train & Test work flow

Extract (not finetuned) and save fc7 features on all Selective Search windows on your dataset (i.e. PASCAL VOC2007) -> Cluster features -> Initial detector training via coverage maximization -> Refine via Smooth Latent SVM -> Test the detectors.

### Usage

[A. Feature extraction on SS windows]
  1. Following the instructions in R-CNN code, precompute fc7 features.
  2. Save features under feat_cache/voc_2007_flip_0_layer_fc7_finetuned_0/

[B. Clustering] This is the slowest and the most computationally expensive step. Suggestions are welcomed. If you want to short cut and get a list of estimated foreground boxes on PASCAL 2007 trainval for all 20 classes. Check `saved_foreground_coords` directory.
  1. On cluster, launch `cluster_patches_parallel_single_nogt_20x1.m` in parallel with inputs (positive image id, classid). The first argument goes from 1 ~ #positive image per class. The second argument goes from 1 ~ 20.
  2. This will save about 70GB of clustering results for PASCAL VOC2007 under directory `paris_results_nogt_20x1/%s_%d.mat` where the first string is for the image id and the second integer if for class id (1~20).

[C. Initial detector training via coverage maximization]
  1. Start matlab.
  2. Launch `train_classes_20x1_smooth_greedycover.m` with input (classid)

[D. Refine via Smooth Latent SVM]
  1. Start matlab
  2. Launch `train_classes_20x1_smooth_lsvm_topK_bagmine_greedycover.m` with input (classid)

[E. Test the detectors]
  1. Start matlab
  2. Launch `eval_classes_20x1_smooth_lsvm_topK_bagmine_greedycover.m` with input (classid)
  3. (optional) to save a nicely parsed table of AP numbers for each class, launch 
`parse_smooth_lsvm_topK_bagmine_greedycover_maxiter_june17.m`.
