% this file produces differents clustering of the houses
% first using Tri's K-mean method
% second using Spectral clustering using the correlation between each house
% third using Spectral clustering using an RBF kernel (exponential of the opposite of
% the L2 distance between each house)
% all clusters are put in the folder "clusters"

%% general settings
nb_clusters=[1,2,4,8,16,32,48,64,128];
pathroot='../';
index=load(strcat(pathroot,'aggregated_data/index.txt'));

% load the real data
nb_line_to_consider=8577;
% we "play" only with 2/3 of the data set


%% the 3 next parts produces the clusters bluid using Tri's k-mean way
%% build the representation of the houses in the 24 dim spaces
% necessitate double loop for -__-

disp('representing each house in a 24 dimensional space');
houses_dim24=zeros(length(index),24);
weights=zeros(length(index),24);
for i=1:length(index)
    disp(strcat('extracting house',num2str(i)));
    house=index(i);
    house_data=load(strcat(pathroot,'aggregated_data/',num2str(house),'.txt'));
    for j=1:nb_line_to_consider
        hour=mod(house_data(j,1),100);
       if hour<=24
           houses_dim24(i,hour)=houses_dim24(i,hour)+house_data(j,2);
           weights(i,hour)=weights(i,hour)+1;
       end
    end
end
%take the mean
houses_dim24=houses_dim24./weights;

%% a small plot, in order to see if there are clear clusters
scatter(houses_dim24(:,8), houses_dim24(:,19));
xlabel('average consumption at 8am');
ylabel('average consumption at 7pm');

%% apply kmean
for nbc=nb_clusters
    disp(strcat('saving clusters :',num2str(nbc)));
    clusters=kmeans(houses_dim24,nbc, 'replicates',100);
    disp(strcat(pathroot,'clusters/trikmeans/',num2str(nbc),'.txt'));
    [fid, message] = fopen(strcat(pathroot,'clusters/trikmeans/',num2str(nbc),'.txt'),'w');
    disp(message);
    for i=1:nbc
       houses_in_clusters=index(clusters==i);
       for j=1:length(houses_in_clusters)
           if j==1
               fprintf(fid, '%d', houses_in_clusters(j));
           else
               fprintf(fid, ',%d', houses_in_clusters(j));
           end
       end
       fprintf(fid, '\n');
    end
    fclose(fid);
end


%% this part build a smilarity matrix that can be used for spectral clustering
% what is important is the way the houses decide to consum 
disp('establish RBF similarity between houses');
RBF_similarity=zeros(length(index));
data_allhouses=zeros(nb_line_to_consider, length(index));
%extract datas

for i=1:length(index)
    disp(strcat('extracting house',num2str(i)));
    house=index(i);
    vect=load(strcat(pathroot,'aggregated_data/',num2str(house),'.txt'));
    data_allhouses(:,i)=vect(end-nb_line_to_consider+1:end);
    data_allhouses(:,i)=(data_allhouses(:,i)-mean(data_allhouses(:,i)))/std(data_allhouses(:,i));
end
% compute RBF similarity
for i=1:length(index)
    i
    house_data=data_allhouses(:,i);
    for j=1:length(index)
        house_data2=data_allhouses(:,j);
        RBF_similarity(i,j)=exp(-mean((house_data-house_data2).^2));
    end
end

%% the 3 other parts realize a spectral clustering
% Every house is a point of a complete graph
% their distance is their cross correlation. 
% Since spectral clustering seem to exige distance between 0 and 1
% All negative values are set to 0
%
load('max_cc_0'); % is calculate in the file correlation_between_house0
S=RBF_similarity;

L=eye(length(index));
for i=1:length(index)
    L(i,i)=sum(S(i,:),2);
end
D=L-S;
[vectors, lambdas]=eig(D); %correl is symetric, thus its eigenvalues are reals
l=diag(lambdas);
[vals, index_eigv]=sort(lambdas); %order lambdas, that is not done by default
%% a small plot between the two first eigen vectors
projected_houses=vectors(:,index_eigv(1:4));
scatter(projected_houses(:,4), projected_houses(:,3));

%% then use kmean
% the previous step has crushed the dats into clusters. Now we can use
% k-mean since the clusters are well defined (more or less)
method='samRBF';
for nbc=nb_clusters
    projected_houses=vectors(:,index_eigv(1:nbc));
    
    disp(strcat('saving clusters :',num2str(nbc)));
    clusters=kmeans(projected_houses, nbc, 'replicates',25);
    disp(strcat(pathroot,'clusters/',method,'/',num2str(nbc),'.txt'));
    [fid, message] = fopen(strcat(pathroot,'clusters/',method,'/',num2str(nbc),'.txt'),'w');
    disp(message);
    for i=1:nbc
       houses_in_clusters=index(clusters==i);
       for j=1:length(houses_in_clusters)
           if j==1
               fprintf(fid, '%d', houses_in_clusters(j));
           else
               fprintf(fid, ',%d', houses_in_clusters(j));
           end
       end
       fprintf(fid, '\n');
    end
    fclose(fid);
end


%% this part exploit the data obtained by clustering
method='trikmeans';

results_rmse=zeros(6,4);
results_mape=zeros(6,4);
type={'linear','mlp','smoreg','avg'};
nb_line_to_consider=4176;

%load the real data
vect=load(strcat(pathroot,'aggregated_data/sum_overall.txt'));
real_consum=vect(end-nb_line_to_consider+1:end);
nb_clusters_studied=[1,2,4,8,16,32,48,64];
for j=1:length(nb_clusters_studied)
    nbc=nb_clusters_studied(j);
    for i=1:4
       vect=load(strcat(pathroot,'clusters/',method,'_predictions/',num2str(nbc),'_clusters/overall_prediction_',type{i},'.txt'));
       predict_consum=vect(end-nb_line_to_consider+1:end);
       results_rmse(j,i)=sqrt(mean((real_consum-predict_consum).^2));
       results_mape(j,i)=mean(abs(real_consum-predict_consum)./real_consum*100);
    end 
end
semilogx(nb_clusters_studied',results_rmse(:,:));
xlabel('number of clusters');
ylabel('RMSE on aggregated consumption kWh');
legend('SVR');
figure
hold all;
semilogx(nb_clusters_studied',results_mape(:,1),'-');
semilogx(nb_clusters_studied',results_mape(:,2),'-s');
semilogx(nb_clusters_studied',results_mape(:,3),'-*');
xlabel('clusters');
ylabel('MAPE on aggregated consumption (kWh)');
legend('linear reg.','SVR','MLP');
