function sen = senpct(df,pct)
%senpct give out the sensitivity of data until the specified percentile
%% Inputs
%
% df         ...   a data the has been sorted by the score of positive
%                  class in descending order
% pct        ...   percentile
%% Output
%
% senpct     ...   sensitivity of the filtered data

    len = length(df);
    index = floor((pct/100)*len);
    fil_data = df(1:index,:);
    conf = confusionmat(fil_data(:,1),fil_data(:,2),'order',[1,0]);
    sen = conf(1,1)/sum(conf(1,:));
end

