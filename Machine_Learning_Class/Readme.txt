========ReadMe========

========Instruction to test the Model========
To test the model against the test set, please use the MATLAB Version R2020a.
1. The testing code is in the file ‘CODE_4_Model_Test.m’
2. You can run it section by section or Run at one time.

========Coding File========
There are 5 coding scripts in this folder. The description of each code is already in
their respective file. The name of the scripts are

Jupyter Notebook
1. 'CODE_0_Data_PreProcessing.ipynb'

Matlab
2. 'CODE_1_Dataset_Preparation.m'
3. 'CODE_2_Decision_Tree_Optimization.m'
4. 'CODE_3_LR_Optimization.m'
5. 'CODE_4_Model_Test.m'

========Function Files=======
Below is the Matlab customised functions built-in specifically for this coursework
1. 'mdllrperf.m'
2. 'mdlperf.m'
3. 'optfitctree.m'
4. 'senpct.m'
5. 'cumres.m'

========.MAT File========
These files contain the dataset and model information in .MAT format
1. 'Tele_Data.mat' --> All the data files after processed in 'CODE_1_Dataset_Preparation.m'
2. 'DT_Model.mat' --> Final set of Decision Tree Models optimised by 
'CODE_2_Decision_Tree_Optimization.m'
3. 'LR_Model.mat' --> Final set of Logistic Regression Models optimised by
'CODE_3_LR_Optimization.m'

=====Dataset Folder=====
The Data original came from https://www.kaggle.com/c/launchds-classification/data
The dataset folder contains dataset files in xlsx format
Please move this file to the same level as the CODE if you wish to run
'CODE_0_Data_PreProcessing.ipynb' or
'CODE_1_Dataset_Preparation.m'

Updated By: Prapas R.
Updated Date: 19/09/2021