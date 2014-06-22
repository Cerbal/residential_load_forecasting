% this file has just been built to show in the report some nice plots of
% average consumption through houses.

%% average overall consum per hour
consums=load('../aggregated_data/sum_overall.txt');
house1002=load('../aggregated_data/1002.txt');
timestamp=mod(house1002(:,1),100);
mean_cons=zeros(1,24);
for i=1:24
    mean_cons(i)=mean(consums(timestamp==i));
end
bar(1:24,mean_cons);
xlabel('hour of the day');
ylabel('average consumption (kWh)');

%% average consum per hour in the week
timestamp=mod(house1002(:,1),100);
timestamp2=mod(floor(house1002(:,1)/100)+2,7);
ts=24*timestamp2+timestamp;
mean_cons=zeros(1,7*24);
for i=1:7*24
    mean_cons(i)=mean(consums(ts==i));
end
plot(1:7*24,mean_cons/782);
xlabel('hour of the week');
ylabel('average overall consumption');

%% this part give a vector with the average consumption of each houses
pathroot='../';
path=strcat(pathroot,'aggregated_data/index.txt');
index=load(path);
consum_houses=zeros(length(index),1);
for i=1:length(index)
    i
   vect=load(strcat(pathroot,'aggregated_data/',num2str(index(i)),'.txt'));
   consum_houses(i)=mean(vect(:,2));
end

%% just for fun, give a week of house 1002 to illustrate that IRL, it is not easy
k=4;
plot(house1002(6*24+1+k*7*24:13*24+k*7*24,2));
xlabel('hour of the week');
ylabel('consumption (kWh)');

%% aggregated consumption for the same week
k=4;
plot(consums(6*24+1+k*7*24:13*24+k*7*24)/782);
xlabel('hour of the week');
ylabel('consumption (kWh)');