%% this script determine what is the best way to discretize the datas :
% load data (training set only)
load('data_test');
data_test=data;
load('data_training');
num_col=8;

% in thos tab, I put the resulting value of discretization
data_dis_unif=zeros(size(data));
data_dis_qtle1=zeros(size(data));
data_dis_qtle2=zeros(size(data));
data_dis_cttm=zeros(size(data));

% same thing but for test set
test_dis_unif=zeros(size(data_test));
test_dis_qtle1=zeros(size(data_test));
test_dis_qtle2=zeros(size(data_test));
test_dis_cttm=zeros(size(data_test));

% this is the tab that contains the cttm discretization, in form of bins
data_training_16bin_cttm=zeros(size(data));

for i=1:1
    i
    col_d=data(:,i);
    col_d_test=data_test(:,i);
    
    % at first uses a uniform sampling
    bounds=linspace(min(col_d),max(col_d),num_col+1)';
    val_interval=(bounds(2:end)+bounds(1:end-1))/2; % take the middle of the interval
                                % , as discretization value
    bounds=bounds(2:end-1);
    data_dis_unif(:,i)=discretize(col_d, bounds, val_interval);
    test_dis_unif(:,i)=discretize(col_d_test,bounds,val_interval);
    
    % then a quantile sampling 
    bounds=quantile(col_d,num_col-1)';
    % for the first version (where we take the middle of the interval)
    bounds_extend=[min(col_d);bounds;max(col_d)];
    val_interval=(bounds_extend(2:end)+bounds_extend(1:end-1))/2; 
    data_dis_qtle1(:,i)=discretize(col_d, bounds, val_interval);
    test_dis_qtle1(:,i)=discretize(col_d_test,bounds,val_interval);
    % for the second version (the value returned is the mean of the
    % interval)
    [~, bins]=histc(col_d, bounds_extend);
    for j=1:num_col
       val_interval(j)= mean(col_d(bins==j));
       if isnan(val_interval(j))
            val_interval(j)=mean(col_d(bins==j+1));
       end
    end
    data_dis_qtle2(:,i)=discretize(col_d, bounds, val_interval);
    test_dis_qtle2(:,i)=discretize(col_d_test,bounds,val_interval);
    
    % finaly discretize accroding to the contribution to the mean
    % (uniform segmentation on the integration)
    sort_col_d=sort(col_d);
    sum_col_d=conv(sort_col_d,ones(length(col_d),1));
    sum_col_d=sum_col_d(1:length(col_d));
    bounds=linspace(min(sum_col_d),max(sum_col_d),num_col+1)';
    [~,bins]=histc(sum_col_d,bounds);
    for j=1:num_col-1
       bounds(j)= max(sort_col_d(bins==j));
    end
    bounds_extend=[-Inf;bounds;Inf];
    [~, bins]=histc(col_d, bounds_extend);
    for j=1:num_col
       val_interval(j)= mean(col_d(bins==j));
    end
    data_dis_cttm(:,i)=discretize(col_d, bounds, val_interval);
    test_dis_cttm(:,i)=discretize(col_d_test,bounds,val_interval);
    [~,data_training_8bin_cttm(:,i)]=histc(col_d,bounds);
end

%% save the discretized data
%save('data_training_8bin_cttm','data_training_8bin_cttm');
%% compute the average rmse for each method
l2diff=zeros(5,size(data,2));
l2diff(1,:)=0;
l2diff(2,:)=sqrt(mean((data-data_dis_unif).^2));
l2diff(3,:)=sqrt(mean((data-data_dis_qtle1).^2));
l2diff(4,:)=sqrt(mean((data-data_dis_qtle2).^2));
l2diff(5,:)=sqrt(mean((data-data_dis_cttm).^2));
mean(l2diff,2)
%% plot result
house_to_plot=1;
plotdata=[data(:,house_to_plot),data_dis_unif(:,house_to_plot),data_dis_qtle1(:,house_to_plot),data_dis_qtle2(:,house_to_plot),data_dis_cttm(:,house_to_plot)];
xbegin=8060,
xfinal=8160;

subplot(3,1,1);
plot((1:size(data,1))',plotdata(:,1),'-');
hold all;
plot((1:size(data,1))',plotdata(:,2),'--');
legend('real', 'uniform');
axis([xbegin,xfinal,0,4]);
xlabel('time (hour)');
ylabel('consumption(kWh)');

% subplot(1,3,2);
% plot((1:size(data,1))',plotdata(:,[1,3]));
% legend('real', 'quantile 1');
% axis([xbegin,xfinal,0,4]);
% xlabel('time in hour');
% ylabel('consumption(kWh)');

subplot(3,1,2);
plot((1:size(data,1))',plotdata(:,1),'-');
hold all;
plot((1:size(data,1))',plotdata(:,4),'--');
legend('real', 'quantile');
axis([xbegin,xfinal,0,4]);
xlabel('time (hour)');
ylabel('consumption(kWh)');

subplot(3,1,3);
plot((1:size(data,1))',plotdata(:,1),'-');
hold all;
plot((1:size(data,1))',plotdata(:,5),'--');
legend('real', 'cttm');
axis([xbegin,xfinal,0,4]);
xlabel('time (hour)');
ylabel('consumption(kWh)');

%% l2diff, but with the test set
l2diff_test=zeros(5,size(data_test,2));
l2diff_test(1,:)=0;
l2diff_test(2,:)=sqrt(mean((data_test-test_dis_unif).^2));
l2diff_test(3,:)=sqrt(mean((data_test-test_dis_qtle1).^2));
l2diff_test(4,:)=sqrt(mean((data_test-test_dis_qtle2).^2));
l2diff_test(5,:)=sqrt(mean((data_test-test_dis_cttm).^2));
mean(l2diff_test,2)