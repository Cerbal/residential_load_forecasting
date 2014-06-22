function [maxcc,lag]=cross_correl(vect1, vect2, lag_max)
    vect1=vect1-mean(vect1);
    vect2=vect2-mean(vect2);
    v1=[zeros(lag_max,1);vect1;zeros(lag_max,1)];

    correl=zeros(2*lag_max,2);
    lags=[-lag_max:-1,1:lag_max];
    for i=1:length(lags)
        correl(i,2)=lags(i);
        correl(i,1)=mean(vect2.*v1(1+lag_max+lags(i):end-lag_max+lags(i)))/std(vect1)/std(vect2);
    end
    [~,indice_max]=max(correl(:,1));
    maxcc=correl(indice_max,1);
    lag=correl(indice_max,2);
end