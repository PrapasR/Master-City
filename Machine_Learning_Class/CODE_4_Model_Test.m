% This script is created to test the best performance model against the
% test data.
%% Section 0: Import Dataset and Model
clear all 
close all
clc

rng('default')
load Tele_Data.mat
load DT_Model.mat
load LR_Model.mat
%% Section 1-1 (Decision Tree): Test Decision Tree Models 

% Extract model performance for Decision Tree Model trained with imbalance
% dataset
[DT_imb_auc,DT_imb_alift,DT_imb_err,DT_imb_sen,Y_imb_resp,X_imb_ROC,Y_imb_ROC]...
    = mdlperf(DT_imb_mdl_final,data_test,'y');

% Extract model performance for Decision Tree Model trained with SMOTE
% dataset
[DT_over_auc,DT_over_alift,DT_over_err,DT_over_sen,Y_over_resp,X_over_ROC,Y_over_ROC]...
    = mdlperf(DT_over_mdl_final,data_test,'y');

% Extract model performance for Decision Tree Model trained with Under
% Random Sampling dataset
[DT_under_auc,DT_under_alift,DT_under_err,DT_under_sen,Y_under_resp,X_under_ROC,Y_under_ROC]...
    = mdlperf(DT_under_mdl_final,data_test,'y');

%% Section 1-2 (Decision Tree): Plot Result in ROC Graph

baselineX = 0:0.1:1;
baselineY = 0:0.1:1;

figure(1)
plot(X_imb_ROC,Y_imb_ROC,'LineWidth',2,'Color',[0.6350 0.0780 0.1840]);
hold on
plot(X_over_ROC,Y_over_ROC,'LineWidth',2,'Color',[0 0.4470 0.7410]);
plot(X_under_ROC,Y_under_ROC,'LineWidth',2,'Color',[0.8500 0.3250 0.0980]);
plot(baselineX,baselineY,'LineWidth',2,'Color','black');
imblgd = sprintf('Imbalance Dataset (AUC values is %.2f)',DT_imb_auc);
overlgd = sprintf('Random Oversampling Dataset (AUC values is %.2f)',DT_over_auc);
underlgd = sprintf('Random Undersampling Dataset (AUC values is %.2f)',DT_under_auc);
legend(imblgd,overlgd,underlgd,'Baseline','Location','southeast');
xlabel('False Positive Rate','FontSize',15);
ylabel('True Positive Rate','FontSize',15);
title('ROC Curves for Decision Tree');
ax = gca;
ax.TitleFontSizeMultiplier = 2;
hold off

%% Section 1-3 (Decision Tree): Plot Result in ALIFT Graph

pct = (0:100)/100;

figure(2)
plot(pct,Y_imb_resp,'LineWidth',2,'Color',[0.6350 0.0780 0.1840]);
hold on
plot(pct,Y_over_resp,'LineWidth',2,'Color',[0 0.4470 0.7410]);
plot(pct,Y_under_resp,'LineWidth',2,'Color',[0.8500 0.3250 0.0980]);
plot(pct,pct,'LineWidth',2,'Color','black');
imblgd = sprintf('Imbalance Dataset (ALIFT values is %.2f)',DT_imb_alift);
overlgd = sprintf('Random Oversampling Dataset (ALIFT values is %.2f)',DT_over_alift);
underlgd = sprintf('Random Undersampling Dataset (ALIFT values is %.2f)',DT_under_alift);
legend(imblgd,overlgd,underlgd,'Baseline','Location','southeast');
xlabel('% of Customer Contacted','FontSize',15); 
ylabel('True Positive Rate','FontSize',15);
title('Cumulative Gain Chart for Decision Tree');
ax = gca;
ax.TitleFontSizeMultiplier = 2;
hold off

%% Section 2-1 (Logistic Regression): Test Logistic Regression Models 

% Extract model performance for Logistic Regression Model trained with imbalance
% dataset
[LR_imb_auc,LR_imb_alift,LR_imb_err,LR_imb_sen,LR_Y_imb_resp,LR_X_imb_ROC,LR_Y_imb_ROC]...
    = mdllrperf(LR_imb_mdl_final,data_test,'y');

% Extract model performance for Logistic Regression Model trained with SMOTE
% dataset
[LR_over_auc,LR_over_alift,LR_over_err,LR_over_sen,LR_Y_over_resp,LR_X_over_ROC,LR_Y_over_ROC]...
    = mdllrperf(LR_over_mdl_final,data_test,'y');

% Extract model performance for Logistic Regression Model trained with Under
% Random Sampling dataset
[LR_under_auc,LR_under_alift,LR_under_err,LR_under_sen,LR_Y_under_resp,LR_X_under_ROC,LR_Y_under_ROC]...
    = mdllrperf(LR_under_mdl_final,data_test,'y');

%% Section 2-2 (Logistic Regression): Plot Result in ROC Graph

baselineX = 0:0.1:1;
baselineY = 0:0.1:1;

figure(3)
plot(LR_X_imb_ROC,LR_Y_imb_ROC,'LineWidth',2,'Color',[0.6350 0.0780 0.1840]);
hold on
plot(LR_X_over_ROC,LR_Y_over_ROC,'LineWidth',2,'Color',[0 0.4470 0.7410]);
plot(LR_X_under_ROC,LR_Y_under_ROC,'LineWidth',2,'Color',[0.8500 0.3250 0.0980]);
plot(baselineX,baselineY,'LineWidth',2,'Color','black');
imblgd = sprintf('Imbalance Dataset (AUC values is %.2f)',LR_imb_auc);
overlgd = sprintf('Random Oversampling Dataset (AUC values is %.2f)',LR_over_auc);
underlgd = sprintf('Random Undersampling Dataset (AUC values is %.2f)',LR_under_auc);
legend(imblgd,overlgd,underlgd,'Baseline','Location','southeast');
xlabel('False Positive Rate','FontSize',15);
ylabel('True Positive Rate','FontSize',15);
title('ROC Curves for Logistic Regression Model');
ax = gca;
ax.TitleFontSizeMultiplier = 2;
hold off

%% Section 2-3 (Logistic Regression): Plot Result in ALIFT Graph

pct = (0:100)/100;

figure(4)
plot(pct,LR_Y_imb_resp,'LineWidth',2,'Color',[0.6350 0.0780 0.1840]);
hold on
plot(pct,LR_Y_over_resp,'LineWidth',2,'Color',[0 0.4470 0.7410]);
plot(pct,LR_Y_under_resp,'LineWidth',2,'Color',[0.8500 0.3250 0.0980]);
plot(pct,pct,'LineWidth',2,'Color','black');
imblgd = sprintf('Imbalance Dataset (ALIFT values is %.2f)',LR_imb_alift);
overlgd = sprintf('Random Oversampling (ALIFT values is %.2f)',LR_over_alift);
underlgd = sprintf('Random Undersampling Dataset (ALIFT values is %.2f)',LR_under_alift);
legend(imblgd,overlgd,underlgd,'Baseline','Location','southeast');
xlabel('% of Customer Contacted','FontSize',15); 
ylabel('True Positive Rate','FontSize',15);
title('Cumulative Gain Chart for Logistic Regression Model');
ax = gca;
ax.TitleFontSizeMultiplier = 2;
hold off

%% Section 3-1 (Appendix): Extract DT Parameter



DT_Model_Name = {'Imbalance Model';'Random Over Sampling Model';'Random Under Sampling Model'};

min_leaf1 = DT_imb_mdl_final.ModelParameters.MinLeaf;
min_leaf2 = DT_over_mdl_final.ModelParameters.MinLeaf;
min_leaf3 = DT_under_mdl_final.ModelParameters.MinLeaf;

max_split1 = DT_imb_mdl_final.ModelParameters.MaxSplits;
max_split2 = DT_over_mdl_final.ModelParameters.MaxSplits;
max_split3 = DT_under_mdl_final.ModelParameters.MaxSplits;

Min_Leaf = [min_leaf1;min_leaf2;min_leaf3];
Max_Split = [max_split1;max_split2;max_split3];

DT_Parameter_Table = table(Min_Leaf,Max_Split,'RowNames',DT_Model_Name)
% writetable(DT_Parameter_Table,'DT_Parameter_Table.xlsx')

%% Section 3-2 (Appendix): Extract LR Parameter



LR_Model_Name = {'Imbalance Model';'Random Over Sampling Model';'Random Under Sampling Model'};

r1 = LR_imb_mdl_final.Rsquared.Ordinary;
r2 = LR_over_mdl_final.Rsquared.Ordinary;
r3 = LR_under_mdl_final.Rsquared.Ordinary;

int1 = table2array(LR_imb_mdl_final.Coefficients(1,1));
int2 = table2array(LR_over_mdl_final.Coefficients(1,1));
int3 = table2array(LR_under_mdl_final.Coefficients(1,1));

RSquared = [r1;r2;r3];
Coef_Intercept = [int1;int2;int3];

LR_Parameter_Table = table(RSquared,Coef_Intercept,'RowNames',LR_Model_Name)