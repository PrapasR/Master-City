function res = cumres(df,pct)
%cumres return the ratio of positive class over the total number of positive class given that the data has been cut to the specified percentile
%(respone can be only 1 for positive class and 0 for negative class)

%% Inputs
%
% df         ...   a data the has been sorted by the score of positive
%                  class in descending order
% pct        ...   percentile
%% Output
%
% res             ...   a ratio of positive class over total number of
%                       positive class

%% Calulate the response ratio
    len = length(df);
    total_pos = sum(df(:,1));
    
    %filter the dataset
    index = floor((pct/100)*len);
    fil_data = df(1:index,:);
    
    %calculate the ratio response
    %conf = confusionmat(fil_data(:,1),fil_data(:,2),'order',[1,0]);
    res = sum(fil_data(:,1))/total_pos;
end

