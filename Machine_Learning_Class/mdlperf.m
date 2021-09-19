function [auc,alift,error,sensitivity,cum_response,XROC,YROC] = mdlperf(mdl,validation_data,target_col)
% A function that return model performance: AUC, ALIFT, error rate, and
% sensitivity
%% Input
%
% mdl                 ...   model
% validation_data     ...   validation data in table format
% target_col          ...   name of target column
%% Output
%
% auc                 ...   AUC value 
% alift               ...   ALIFT value 
% error               ...   Error value 
% sensitivity         ...   Sensitivity value 
% cum_response        ...   cumulative response of ratio between positive class over the total number of positive class
% XROC                ...   X axis coordinate of ROC Curve
% YROC                ...   Y axis coordinate of ROC Curve

%%
 error = loss(mdl,validation_data,target_col);
 
 [ypred,ypredscore] = predict(mdl,validation_data);
 val_data_compare = [validation_data.y ypred ypredscore(:,2)];
 sorted_val_data_compare = sortrows(val_data_compare,3,'descend');
 sensitivity = senpct(sorted_val_data_compare,100);
 
 [XROC,YROC,~,auc] = perfcurve(validation_data.y,ypredscore(:,2),true);
 
 pct = 0:100;
 cum_response = [];
 for i = pct
     resp = cumres(sorted_val_data_compare,i);
     cum_response = [cum_response resp];
 end
 
 alift =trapz(pct/100,cum_response);

end

