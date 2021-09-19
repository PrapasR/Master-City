% This script is used to optimize the logistic regression model. In this
% script, 3 models are trained with original imbalance data, random over sampling dataset
% dataset and random under sampling dataset.
%% Section 0: Import Dataset
clear all 
close all
clc

rng('default')

load Tele_Data.mat;
target_col = 'y';

%% Section 1: Train Logistic Regression for each dataset (Set Linear Interaction between variables)

LR_imb_mdl = fitglm(data_train_imbalance,'Distribution','binomial');
LR_over_mdl = fitglm(data_train_over,'Distribution','binomial');
LR_under_mdl = fitglm(data_train_under,'Distribution','binomial');

%LR_smote_mdl = fitglm(data_train_smote,'Distribution','binomial');
%% Section 2: Train Logistic Regression for each dataset using stepwise (Set Linear Interaction)
LR__stp_imb_mdl = stepwiseglm(data_train_imbalance,'linear','upper','linear','Distribution','binomial');
LR__stp_over_mdl = stepwiseglm(data_train_over,'linear','upper','linear','Distribution','binomial');
LR__stp_under_mdl = stepwiseglm(data_train_under,'linear','upper','linear','Distribution','binomial');

%LR__stp_smote_mdl = stepwiseglm(data_train_smote,'linear','upper','linear','Distribution','binomial');
%% Section 3: Extract the results

% 1 = LR_imb_mdl
[auc_1,alift_1,error_1,sensitivity_1] = mdllrperf(LR_imb_mdl,data_validation,target_col);

% 2 = LR_over_mdl
[auc_2,alift_2,error_2,sensitivity_2] = mdllrperf(LR_over_mdl,data_validation,target_col);

% 3 = LR_under_mdl
[auc_3,alift_3,error_3,sensitivity_3] = mdllrperf(LR_under_mdl,data_validation,target_col);

% 4 = LR__stp_imb_mdl
[auc_4,alift_4,error_4,sensitivity_4] = mdllrperf(LR__stp_imb_mdl,data_validation,target_col);

% 5 = LR__stp_over_mdl
[auc_5,alift_5,error_5,sensitivity_5] = mdllrperf(LR__stp_over_mdl,data_validation,target_col);

% 6 = LR__stp_under_mdl
[auc_6,alift_6,error_6,sensitivity_6] = mdllrperf(LR__stp_under_mdl,data_validation,target_col);



adjR2_1 = LR_imb_mdl.Rsquared.Adjusted;
adjR2_2 = LR_over_mdl.Rsquared.Adjusted;
adjR2_3 = LR_under_mdl.Rsquared.Adjusted;
adjR2_4 = LR__stp_imb_mdl.Rsquared.Adjusted;
adjR2_5 = LR__stp_over_mdl.Rsquared.Adjusted;
adjR2_6 = LR__stp_under_mdl.Rsquared.Adjusted;
%% Section 4: Evaluate the Logistic Models
Model_Name = {'LR_imb_mdl';'LR_over_mdl';'LR_under_mdl';...
    'LR__stp_imb_mdl';'LR__stp_over_mdl';'LR__stp_under_mdl'};
AUC = [auc_1;auc_2;auc_3;auc_4;auc_5;auc_6];
ALIFT = [alift_1;alift_2;alift_3;alift_4;alift_5;alift_6];
Error = [error_1;error_2;error_3;error_4;error_5;error_6];
Sensitivity = [sensitivity_1;sensitivity_2;sensitivity_3;sensitivity_4;sensitivity_5;sensitivity_6];
AdjR2 = [adjR2_1;adjR2_2;adjR2_3;adjR2_4;adjR2_5;adjR2_6];


LR_Result_Table = table(AUC,ALIFT,Error,Sensitivity,AdjR2,'RowNames',Model_Name);
writetable(LR_Result_Table,'LR_Result_Table.xlsx');

%% Section X-X: Export models
% Export model trained by original imbalance data, random over sampling
% dataset and random under sampling dataset.

% The performance of logistic regression models trained by fitglm and stepwiseglm
% are not difference from each other. The models trained from stepwiseglm
% are selected to be the final model since the model already remove the
% unsignificant variables and is more suitable to interpret to the users.

LR_imb_mdl_final = LR__stp_imb_mdl;
LR_over_mdl_final = LR__stp_over_mdl;
LR_under_mdl_final = LR__stp_under_mdl;

save('LR_Model.mat',...
    'LR_imb_mdl_final','LR_over_mdl_final','LR_under_mdl_final');



%% Section X-1: Assume that the model has some interaction between predictor value

%The test between Logistic Regression model that have interaction between
%each predictor has been left out due to computationally expensive from
%using stepwiseglm function.


% LR_imb_interact_mdl = fitglm(data_train_imbalance,'interactions','Distribution','binomial');
% LR_over_interact_mdl = fitglm(data_train_over,'interactions','Distribution','binomial');
% LR_smote_interact_mdl = fitglm(data_train_smote,'interactions','Distribution','binomial');
% LR_under_interact_mdl = fitglm(data_train_under,'interactions','Distribution','binomial');

%% Section X-2: Assume that the model has some interaction between predictor value
% LR__stp_imb_interact_mdl = stepwiseglm(data_train_imbalance,'linear','upper','interactions','Distribution','binomial');
% LR__stp_over_interact_mdl = stepwiseglm(data_train_over,'linear','upper','interactions','Distribution','binomial');
% LR__stp_under_interact_mdl = stepwiseglm(data_train_under,'linear','upper','interactions','Distribution','binomial');

%% Section X-3: Extract the results

% 7 = LR_imb_interact_mdl
% [auc_7,alift_7,error_7,sensitivity_7] = mdllrperf(LR_imb_interact_mdl,data_validation,target_col);

% 8 = LR_smote_interact_mdl
% [auc_8,alift_8,error_8,sensitivity_8] = mdllrperf(LR_over_interact_mdl,data_validation,target_col);

% 9 = LR_under_interact_mdl
% [auc_9,alift_9,error_9,sensitivity_9] = mdllrperf(LR_under_interact_mdl,data_validation,target_col);

% 10 = LR__stp_imb_interact_mdl
% [auc_10,alift_10,error_10,sensitivity_10] = mdllrperf(LR__stp_imb_interact_mdl,data_validation,target_col);

% 11 = LR__stp_smote_interact_mdl
% [auc_11,alift_11,error_11,sensitivity_11] = mdllrperf(LR__stp_smote_interact_mdl,data_validation,target_col);
% 
% 12 = LR__stp_under_interact_mdl
% [auc_12,alift_12,error_12,sensitivity_12] = mdllrperf(LR__stp_under_interact_mdl,data_validation,target_col);

