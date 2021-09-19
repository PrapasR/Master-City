% This script is used to read in the pre-processed data in the excel format and
% convert it to the correct data type

%% Section 0: Initial setup and data read in
% Set Seed Number
rng('default')
%Import training dataset 
data_train_imbalance = readtable('tele_train.xlsx','ReadRowNames',true);
data_train_under = readtable('tele_train_UnderSampling.xlsx','ReadRowNames',true);
data_train_over = readtable('tele_train_OverSampling.xlsx','ReadRowNames',true);
%data_train_smote = readtable('tele_train_smote.xlsx','ReadRowNames',true);

%Import Validaiton dataset
data_validation = readtable('tele_validation.xlsx','ReadRowNames',true);

%Import testing dataset
data_test = readtable('tele_test.xlsx','ReadRowNames',true);

%% Section 1: Change the data type of imbalance dataset
data_train_imbalance.job = categorical(data_train_imbalance.job);
data_train_imbalance.marital = categorical(data_train_imbalance.marital);
data_train_imbalance.education = categorical(data_train_imbalance.education);
data_train_imbalance.default = categorical(data_train_imbalance.default);
data_train_imbalance.housing = categorical(data_train_imbalance.housing);
data_train_imbalance.loan = categorical(data_train_imbalance.loan);
data_train_imbalance.contact = categorical(data_train_imbalance.contact);
data_train_imbalance.month = categorical(data_train_imbalance.month);
data_train_imbalance.day_of_week = categorical(data_train_imbalance.day_of_week);
data_train_imbalance.poutcome = categorical(data_train_imbalance.poutcome);
data_train_imbalance.age_bins = categorical(data_train_imbalance.age_bins);
data_train_imbalance.y = logical(data_train_imbalance.y);

%% Section 2:Change the data type of  Undersampling dataset
data_train_under.job = categorical(data_train_under.job);
data_train_under.marital = categorical(data_train_under.marital);
data_train_under.education = categorical(data_train_under.education);
data_train_under.default = categorical(data_train_under.default);
data_train_under.housing = categorical(data_train_under.housing);
data_train_under.loan = categorical(data_train_under.loan);
data_train_under.contact = categorical(data_train_under.contact);
data_train_under.month = categorical(data_train_under.month);
data_train_under.day_of_week = categorical(data_train_under.day_of_week);
data_train_under.poutcome = categorical(data_train_under.poutcome);
data_train_under.age_bins = categorical(data_train_under.age_bins);
data_train_under.y = logical(data_train_under.y);

%% Section 3:Change the data type of Oversampling dataset
data_train_over.job = categorical(data_train_over.job);
data_train_over.marital = categorical(data_train_over.marital);
data_train_over.education = categorical(data_train_over.education);
data_train_over.default = categorical(data_train_over.default);
data_train_over.housing = categorical(data_train_over.housing);
data_train_over.loan = categorical(data_train_over.loan);
data_train_over.contact = categorical(data_train_over.contact);
data_train_over.month = categorical(data_train_over.month);
data_train_over.day_of_week = categorical(data_train_over.day_of_week);
data_train_over.poutcome = categorical(data_train_over.poutcome);
data_train_over.age_bins = categorical(data_train_over.age_bins);
data_train_over.y = logical(data_train_over.y);



%% Section 4:Change the data type of validaiton dataset
data_validation.job = categorical(data_validation.job);
data_validation.marital = categorical(data_validation.marital);
data_validation.education = categorical(data_validation.education);
data_validation.default = categorical(data_validation.default);
data_validation.housing = categorical(data_validation.housing);
data_validation.loan = categorical(data_validation.loan);
data_validation.contact = categorical(data_validation.contact);
data_validation.month = categorical(data_validation.month);
data_validation.day_of_week = categorical(data_validation.day_of_week);
data_validation.poutcome = categorical(data_validation.poutcome);
data_validation.age_bins = categorical(data_validation.age_bins);
data_validation.y = logical(data_validation.y);

%% Section 5:Change the data type of test
data_test.job = categorical(data_test.job);
data_test.marital = categorical(data_test.marital);
data_test.education = categorical(data_test.education);
data_test.default = categorical(data_test.default);
data_test.housing = categorical(data_test.housing);
data_test.loan = categorical(data_test.loan);
data_test.contact = categorical(data_test.contact);
data_test.month = categorical(data_test.month);
data_test.day_of_week = categorical(data_test.day_of_week);
data_test.poutcome = categorical(data_test.poutcome);
data_test.age_bins = categorical(data_test.age_bins);
data_test.y = logical(data_test.y);



%% Section 6: Export the data
save('Tele_Data.mat',...
    'data_train_imbalance','data_train_under','data_train_over',...
    'data_validation','data_test')

%% Change data type of smote
% data_train_smote.job = categorical(data_train_smote.job);
% data_train_smote.marital = categorical(data_train_smote.marital);
% data_train_smote.education = categorical(data_train_smote.education);
% data_train_smote.default = categorical(data_train_smote.default);
% data_train_smote.housing = categorical(data_train_smote.housing);
% data_train_smote.loan = categorical(data_train_smote.loan);
% data_train_smote.contact = categorical(data_train_smote.contact);
% data_train_smote.month = categorical(data_train_smote.month);
% data_train_smote.day_of_week = categorical(data_train_smote.day_of_week);
% data_train_smote.poutcome = categorical(data_train_smote.poutcome);
% data_train_smote.age_bins = categorical(data_train_smote.age_bins);
% data_train_smote.y = logical(data_train_smote.y);