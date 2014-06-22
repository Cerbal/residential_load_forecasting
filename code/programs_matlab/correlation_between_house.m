%% this file establish the matrix of cross correlation between houses
% for each houses, we establish the correlation between each house
% for a time lag varying from 
path='../aggregated_data/index.txt';
index=load(path);
nb_line_to_consider=8577; % we "play" only with 2/3 of the data set
%% first part, we import the set of data 
data=zeros(nb_line_to_consider,length(index));
for i=1:length(index)
    i
    house=index(i);
    house_data=load(strcat('../aggregated_data/',num2str(house),'.txt'));
    data(:,i)=house_data(1:8577,2);
end
disp('importation of the data of houses done');

%% second part, we establish correlation
max_cc=zeros(length(index));
max_lag=zeros(length(index));
for i=1:length(index)
    i
    for j=i:length(index)
        [a,b]=cross_correl(data(:,i),data(:,j),4);
        max_cc(i,j)=a;
        max_lag(i,j)=b;
    end
end
%% save the result
save('max_cc','max_cc');
save('max_lag','max_lag');

%% export a txt file, that indicate for each house the four best leader we have found for it
% more precisely we give the 5 best leaders that we found
load('max_cc');
load('max_lag');
max_cc=max_cc+max_cc';
for i=1:size(max_cc,1)
   max_cc(i,i)=max_cc(i,i)/2+1; % note, we artificially increase self correlation of 1,
   % so when the correlations will be sorted, the house itself will come at
   % first leader
end
max_lag=max_lag-max_lag'+eye(size(max_cc,1));
disp('ok')

nb_leaders=5;
leaders=zeros(length(index),nb_leaders*3);

for i=1:length(index);
    i
   values=[max_cc(i,:)',max_lag(i,:)',(1:length(index))'];
   values=sortrows(values,-1);
   values=values(values(:,2)>0,:);
   values(1,1)=values(1,1)-1; % after sorting, we substract 1 to the correlation of the house 
   %itself, so this value is now the real correlation
   count=0;
   while (count<nb_leaders && count<size(values,1))
        leaders(i,3*count+1)=index(values(count+1,3));
        leaders(i,3*count+2)=values(count+1,2);
        leaders(i,3*count+3)=values(count+1,1);
        count=count+1;
   end
end

%% export the leader
dlmwrite('leaders.txt',leaders, 'precision','%.3f');
leaders_correl=leaders;
save('leaders_correl','leaders_correl');
%% exploitation
%plot the correlation with the first leader vs the correlation with itself
%delayed
figure;
load('leaders_correl');
leaders=leaders_correl;
plot(leaders(:,3),leaders(:,6),'+');
hold all;
plot([0,1],[0,1]);
xlabel('correlation with itself with lag 1');
ylabel('correlation with the first external leader');
[mean(leaders(:,3)),mean(leaders(:,6))]