function [auc_val,alift_val,err_val,sen_val] = optfitctree(train_data,target_col,validation_data,nleaf,nsplit,varargin)
% A function that iterate through the combination of number of minimim leaf
% and maximum split to using fitctree using curvature prediction
%% Input
%
% train_data          ...   trainning data in table format
% validation_data     ...   validation data in table format
% target_col          ...   name of target column
% nleaf               ...   upper limit number of minimum leaf 
% nsplit              ...   upper limit number of maximum split 
%% Output
%
% auc_val             ...   AUC value for each leaf and split combination
% alift_val           ...   ALIFT value for each leaf and split combination
% err_val             ...   Error value for each leaf and split combination
% sen_val             ...   Sensitivity value for each leaf and split combination

%% Optional (named) Inputs
% 'splitcrit'         ...   split criterion for deciiton tree
% 'rate'              ...   learning rate for number of number of minimum
%                           leaf and maximum split

%% Parse inputs

[splitcrit, rate] = process_options(varargin, ...
    'splitcrit' , 'gdi' , ...
    'rate'  , 1         )  


%% Initialize
%  Setting up the number minimum leaf, maximum, split, and container for
%  output values

% Prepare range of parameter
leafs = 1:rate:nleaf;
maxsplit = 2:rate:nsplit;

% temporary container 
Nleaf = numel(leafs);
Nsplit = numel(maxsplit);

%Container for output
auc_val = zeros(Nleaf,Nsplit);
alift_val = zeros(Nleaf,Nsplit);
err_val = zeros(Nleaf,Nsplit);
sen_val = zeros(Nleaf,Nsplit);



for s = 1:Nsplit
    for n = 1:Nleaf
        tree_mdl = fitctree(train_data,target_col,'PredictorSelection','curvature','SplitCriterion',splitcrit,'MinLeafSize',leafs(n),'MaxNumSplits',maxsplit(s));
        
        % Calulate error rate
        err_val(n,s) = loss(tree_mdl,validation_data,target_col);
        
        % Prepare sorted data that sort probability score by descent
        [ypred,ypredscore] = predict(tree_mdl,validation_data); 
        val_data_compare = [validation_data.y ypred ypredscore(:,2)];
        sorted_val_data_compare = sortrows(val_data_compare,3,'descend');
        
        % Calulate sensitivity rate
        sen_val(n,s) = senpct(sorted_val_data_compare,100);
    
        % Calculate ROC measure from cossvalidation
        % Code from https://www.mathworks.com/help/stats/perfcurve.html#responsive_offcanvas
        [~,~,~,auc] = perfcurve(validation_data.y,ypredscore(:,2),true);
        auc_val(n,s) = auc;
    

        
        % Calculate cumulative response by percentile
        pct = 1:100;
        cum_response = [];
        for i = pct
            resp = cumres(sorted_val_data_compare,i);
            cum_response = [cum_response resp];
        end
        alift_val(n,s) =trapz(pct/100,cum_response);
        n,s
    end  
end
end

