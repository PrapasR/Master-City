% This script is used to optimize the decision tree model. In this
% script, 3 models are trained with original imbalance data, random over sampling dataset
% dataset and random under sampling dataset. The three model were tunned by
% varying the minimum number of tree and maximum number of split and test
% against the validation dataset.

%% Section 0: Import Dataset
clear all 
close all
clc

rng('default')
load Tele_Data.mat
%% Section 1-1 (Imbalance dataset): Find the range of hyperparameter
% The range of hyperparameter was initially created by matlab optimisation
% function

tree_imbalance_mdl_Opt_hyper = fitctree(data_train_imbalance,'y','PredictorSelection','curvature','OptimizeHyperparameters','all')

imbalance_minleaf = tree_imbalance_mdl_Opt_hyper.ModelParameters.MinLeaf;
imbalance_maxsplit = tree_imbalance_mdl_Opt_hyper.ModelParameters.MaxSplits;
imbalance_splitcrit = tree_imbalance_mdl_Opt_hyper.ModelParameters.SplitCriterion;

%%  Section 1-2 (Imbalance dataset): Extract the initial model Performance
%Extract the model performance that had trained by Matlab built-in function
[DT_builtin_auc,DT_builtin_alift,DT_builtin_err,DT_builtin_sen] = mdlperf(tree_imbalance_mdl_Opt_hyper,data_validation,'y');

%%  Section 1-3 (Imbalance dataset): Model Tuning
% Number of optimum number of leaf and maximum split from using matlab
% built in function has been mutiply. This is done so that the model performance can be clearly identified. 
% For reproducibility uncomment next line
%imbalance_minleaf = 19; imbalance_maxsplit = 3;imbalance_splitcrit = 'deviance';
target_col = 'y';
split_mutiplier = 8;
leaf_mutiplier =8;
rate=2

[DT_imb_auc_arr,DT_imb_alift_arr,DT_imb_err_arr,DT_imb_sen_arr] = ...
    optfitctree(data_train_imbalance,target_col,data_validation,...
    imbalance_minleaf*leaf_mutiplier,imbalance_maxsplit*split_mutiplier,...
    'splitcrit',imbalance_splitcrit,'rate',rate);

%%  Section 1-4 (Imbalance dataset): Check the result of AUC and ALIFT
% Visualised AUC and ALIFT
figure(1) 
surf_alift_1 = surf(DT_imb_alift_arr,'FaceColor',[0 0.4470 0.7410]);
xlabel('Maximum Split');
ylabel('Leaf');
zlabel('ALIFT');

figure(2)
surf_auc_1 = surf(DT_imb_auc_arr,'FaceColor',[0.9290 0.6940 0.1250]);
xlabel('Maximum Split');
ylabel('Leaf');
zlabel('AUC');

%% Section 1-5 (Imbalance dataset): Extract the best hyperparameters and train the final model
% From the result of AUC and ALIFT values, the model
% show significant improvement at the minimum leaf position 76 and at maximum
% split position 9
% Note: The number of leaf and split also show significant improvement at
% various location, but this set of optimum parameters has been select to
% decrease the complexity of the model (low maximum split with not too small number of minimum leaf).

imb_leaf = 1:rate:imbalance_minleaf*leaf_mutiplier;
imb_leaf = imb_leaf(76);

imb_split = 2:rate:imbalance_maxsplit*split_mutiplier;
imb_split = imb_split(9);

DT_imb_mdl = fitctree(data_train_imbalance,target_col,'PredictorSelection','curvature','SplitCriterion',imbalance_splitcrit,'MinLeafSize',imb_leaf,'MaxNumSplits',imb_split);

[DT_imb_auc,DT_imb_alift,DT_imb_err,DT_imb_sen] = mdlperf(DT_imb_mdl,data_validation,'y');

%% Section 2-1 (Random Over Sampling dataset): Find the range of hyperparameter
% The range of hyperparameter was initially created by matlab optimisation
% function

tree_over_mdl_Opt_hyper = fitctree(data_train_over,'y','PredictorSelection','curvature','OptimizeHyperparameters','all')

over_minleaf = tree_over_mdl_Opt_hyper.ModelParameters.MinLeaf;
over_maxsplit = tree_over_mdl_Opt_hyper.ModelParameters.MaxSplits;
over_splitcrit = tree_over_mdl_Opt_hyper.ModelParameters.SplitCriterion;
%%  Section 2-2 (Random Over Sampling dataset): Extract the initial model Performance
%Extract the model performance that had trained by Matlab built-in function
[DT_builtin_over_auc,DT_builtin_over_alift,DT_builtin_over_err,DT_builtin_over_sen] = mdlperf(tree_over_mdl_Opt_hyper,data_validation,'y');

%%  Section 2-3 (Random Over Sampling dataset): Model Tuning
% The number of hyperparameters given by the built-in is very deep and
% complex; Number of minimum leaf is 1 and maximum split of more than
% 8,000
% Hence,the new parameters range is defined to avoid computation expensive
% and overfitting
target_col = 'y';
leaf_over = 40;
max_split = 800;
rate=5;

[DT_over_auc_arr,DT_over_alift_arr,DT_over_err_arr,DT_over_sen_arr] = ...
    optfitctree(data_train_over,target_col,data_validation,...
    leaf_over,max_split,...
    'splitcrit',over_splitcrit,'rate',rate);
%%  Section 2-4 (Random Over Sampling dataset): Check the result of AUC and ALIFT
% Visualised AUC and ALIFT
figure(3)
surf_alift_2 = surf(DT_over_alift_arr,'FaceColor',[0 0.4470 0.7410]);
xlabel('Maximum Split');
ylabel('Leaf');
zlabel('ALIFT');

figure(4)
surf_auc_2 = surf(DT_over_auc_arr,'FaceColor',[0.9290 0.6940 0.1250]);
xlabel('Maximum Split');
ylabel('Leaf');
zlabel('AUC');

%% Section 2-5 (Random Over Sampling dataset): Extract the best hyperparameters and train the final model
% From the result of AUC and ALIFT values, the model
% show significant improvement at the minimum leaf position 7 and at maximum
% split position 160
% Note: The number of leaf and split also show significant improvement at
% various location, but this set of  optimum parameters has been select to
% decrease the complexity of the model (low maximum split with not too small number of minimum leaf).

over_leaf = 1:rate:leaf_over;
over_leaf = over_leaf(7);

over_split = 2:rate:max_split;
over_split = over_split(160);

DT_over_mdl = fitctree(data_train_over,target_col,'PredictorSelection','curvature','SplitCriterion',over_splitcrit,'MinLeafSize',over_leaf,'MaxNumSplits',over_split);

[DT_over_auc,DT_over_alift,DT_over_err,DT_over_sen] = mdlperf(DT_over_mdl,data_validation,'y');

%% Section 3-1 (Random Under Sampling dataset): Find the range of hyperparameter
% The range of hyperparameter was initially created by matlab optimisation
% function

tree_under_mdl_Opt_hyper = fitctree(data_train_under,'y','PredictorSelection','curvature','OptimizeHyperparameters','all')

under_minleaf = tree_under_mdl_Opt_hyper.ModelParameters.MinLeaf;
under_maxsplit = tree_under_mdl_Opt_hyper.ModelParameters.MaxSplits;
under_splitcrit = tree_under_mdl_Opt_hyper.ModelParameters.SplitCriterion;
%%  Section 3-2 (Random Under Sampling dataset): Extract the initial model Performance
%Extract the model performance that had trained by Matlab built-in function
[DT_builtin_under_auc,DT_builtin_under_alift,DT_builtin_under_err,DT_builtin_under_sen] = mdlperf(tree_under_mdl_Opt_hyper,data_validation,'y');

%%  Section 3-3 (Random Under Sampling dataset): Model Tuning
% Number of optimum number of leaf and maximum split from using matlab 
% built in function has been mutiply. This is done so that the model performance can be clearly identified. 
% For reproducibility uncomment the below line
%under_minleaf = 1 ; under_maxsplit = 1ุุ6 ;  under_splitcrit = 'deviance';
target_col = 'y';
split_mutiplier = 2;
leaf_mutiplier =100;
rate=2

[DT_under_auc_arr,DT_under_alift_arr,DT_under_err_arr,DT_under_sen_arr] = ...
    optfitctree(data_train_under,target_col,data_validation,...
    under_minleaf*leaf_mutiplier,under_maxsplit*split_mutiplier,...
    'splitcrit',under_splitcrit,'rate',rate);

%%  Section 3-4 (Random Under Sampling dataset): Check the result of AUC and ALIFT
% Visualised AUC and ALIFT
figure(4) 
surf_alift_3 = surf(DT_under_alift_arr,'FaceColor',[0 0.4470 0.7410]);
xlabel('Maximum Split');
ylabel('Leaf');
zlabel('ALIFT');

figure(5)
surf_auc_3 = surf(DT_under_auc_arr,'FaceColor',[0.9290 0.6940 0.1250]);
xlabel('Maximum Split');
ylabel('Leaf');
zlabel('AUC');

%% Section 3-5 (Random Under Sampling dataset): Extract the best hyperparameters and train the final model
% From the result of AUC and ALIFT values, the model
% show significant improvement at the minimum leaf position 7 and at maximum
% split position 8
% Note: The number of leaf and split also show significant improvement at
% various location, but this set of  optimum parameters has been select to
% decrease the complexity of the model (low maximum split with not too small number of minimum leaf).

under_leaf = 1:rate:under_minleaf*leaf_mutiplier;
under_leaf = under_leaf(7);

under_split = 2:rate:under_maxsplit*split_mutiplier;
under_split = under_split(8);

DT_under_mdl = fitctree(data_train_under,target_col,'PredictorSelection','curvature','SplitCriterion',under_splitcrit,'MinLeafSize',under_leaf,'MaxNumSplits',under_split);

[DT_under_auc,DT_under_alift,DT_under_err,DT_under_sen] = mdlperf(DT_under_mdl,data_validation,'y');

%% Section 4-1: Evaluate the models

Model_Name = {'DT_builtin_imb_mdl';'DT_imb_mdl';'DT_builtin_over_mdl';'DT_over_mdl';...
    'DT_builtin_under_mdl';'DT_under_mdl'};
AUC = [DT_builtin_auc;DT_imb_auc;DT_builtin_over_auc;DT_over_auc;DT_builtin_under_auc;DT_under_auc];
ALIFT = [DT_builtin_alift;DT_imb_alift;DT_builtin_over_alift;DT_over_alift;DT_builtin_under_alift;DT_under_alift];
Error = [DT_builtin_err;DT_imb_err;DT_builtin_over_err;DT_over_err;DT_builtin_under_err;DT_under_err];
Sensitivity = [DT_builtin_sen;DT_imb_sen;DT_builtin_over_sen;DT_over_sen;DT_builtin_under_sen;DT_under_sen];

DT_Result_Table = table(AUC,ALIFT,Error,Sensitivity,'RowNames',Model_Name);
writetable(DT_Result_Table,'DT_Result_Table.xlsx')




%% Section 4-2: Export models
% Export model trained by original imbalance data, random over sampling
% dataset and random under sampling dataset.
% The grid seach model of imbalance and under sampling outperform the
% built-in matlab. Only for the oversampling model that built-in matlab
% function outperform.

% Import best models for each type of training set data

DT_imb_mdl_final = DT_imb_mdl;
DT_over_mdl_final = tree_over_mdl_Opt_hyper;
DT_under_mdl_final = DT_under_mdl;


save('DT_Model.mat',...
    'DT_imb_mdl_final','DT_over_mdl_final','DT_under_mdl_final');


%% Section X-1 (SMOTE dataset): Find the range of hyperparameter
% The range of hyperparameter was initially created by matlab optimisation
% % function
% 
% tree_smote_mdl_Opt_hyper = fitctree(data_train_smote,'y','PredictorSelection','curvature','OptimizeHyperparameters','all')
% 
% smote_minleaf = tree_smote_mdl_Opt_hyper.ModelParameters.MinLeaf;
% smote_maxsplit = tree_smote_mdl_Opt_hyper.ModelParameters.MaxSplits;
% smote_splitcrit = tree_smote_mdl_Opt_hyper.ModelParameters.SplitCriterion;
%%  Section X-2 (SMOTE dataset): Extract the initial model Performance
%Extract the model performance that had trained by Matlab built-in function
% [DT_builtin_smote_auc,DT_builtin_smote_alift,DT_builtin_smote_err,DT_builtin_smote_sen] = mdlperf(tree_smote_mdl_Opt_hyper,data_validation,'y');

%%  Section X-3 (Imbalance dataset): Model Tuning
% The number of hyperparameters given by the built-in is very deep and complex. 
% Hence,the new parameters range is defined to avoid computation expensive
% target_col = 'y';
% leaf_smote = 100;
% max_split = 100;
% rate=2;
% 
% [DT_smote_auc_arr,DT_smote_alift_arr,DT_smote_err_arr,DT_smote_sen_arr] = ...
%     optfitctree(data_train_smote,target_col,data_validation,...
%     leaf_smote,max_split,...
%     'splitcrit',smote_splitcrit,'rate',rate);
% %%  Section 2-4 (SMOTE dataset): Check the result of AUC and ALIFT
% % Visualised AUC and ALIFT
% figure(3) 
% surf_alift = surf(DT_smote_alift_arr,'FaceColor',[0 0.4470 0.7410]);
% xlabel('Maximum Split');
% ylabel('Leaf');
% zlabel('ALIFT');
% 
% figure(4)
% surf_auc = surf(DT_smote_auc_arr,'FaceColor',[0.9290 0.6940 0.1250]);
% xlabel('Maximum Split');
% ylabel('Leaf');
% zlabel('AUC');

%% Section X-5 (SMOTE dataset): Extract the best hyperparameters
% From the result of AUC and ALIFT values, the model
% show significant improvement at the minimum leaf position 22 and at maximum
% split position 24
% Note: The number of leaf and split also show significant improvement at
% various location, but this set of  optimum parameters has been select to
% decrease the complexity of the model (low maximum split with not too small number of minimum leaf).

% smote_leaf = 1:rate:leaf_smote;
% smote_leaf = smote_leaf(22);
% 
% smote_split = 2:rate:max_split;
% smote_split = smote_split(24);
% 
% DT_smote_mdl = fitctree(data_train_smote,target_col,'PredictorSelection','curvature','SplitCriterion',smote_splitcrit,'MinLeafSize',smote_leaf,'MaxNumSplits',smote_split);
% 
% [DT_smote_auc,DT_smote_alift,DT_smote_err,DT_smote_sen] = mdlperf(DT_smote_mdl,data_validation,'y');