function [auc,alift,error,sensitivity,cum_response,XROC,YROC] = mdllrperf(mdl,validation_data,target_col)
% A function that return model performance: AUC, ALIFT, error rate, and
% sensitivity
%% Input
%
% mdl                 ...   Logistic Regresion model
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
  
 ypredscore = predict(mdl,validation_data);
 ypred = ypredscore;
 ypred(ypred >=0.5) = 1;
 ypred(ypred <0.5) = 0;
 
 nanidx = find(isnan(ypred));
 
 ypredscore(nanidx, :) =[];
 ypred(nanidx, : ) = [];
 validation_data(nanidx, : ) = [];
 
 ypred = logical(ypred);
 val_data_compare = [validation_data.y ypred ypredscore];
 sorted_val_data_compare = sortrows(val_data_compare,3,'descend');

 sensitivity = senpct(sorted_val_data_compare,100);
 [XROC,YROC,~,auc] = perfcurve(validation_data.y,ypredscore,true);
 
 len_val = length(ypred);
 
 cor_pred = sorted_val_data_compare(:,1) ==sorted_val_data_compare(:,2);
 error = sum(cor_pred)/len_val;
 
 
 

 pct = 0:100;
 cum_response = [];
 for i = pct
     resp = cumres(sorted_val_data_compare,i);
     cum_response = [cum_response resp];
 end
 
 alift =trapz(pct/100,cum_response);

end

