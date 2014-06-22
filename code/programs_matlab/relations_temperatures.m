%% establish the relation between temperature and comsuption
path_root='../';
temps=load(strcat(path_root,'temperatures/temperatures.txt'));
consums=load('../aggregated_data/sum_overall.txt');
data=[consums,temps];
scatter(consums,temps)
xlabel('overall consumption (kWh)');
ylabel('temperature (°C)');

%% relation temp at 20h
figure;
consums=load('../aggregated_data/sum_overall.txt');
house1002=load('../aggregated_data/1002.txt');
timestamp=mod(house1002(:,1),100);
cons_20h=consums(timestamp==19)/782;
temps_20h=temps(timestamp==19);
scatter(cons_20h,temps_20h)
xlabel('overall consumption at 7pm(kWh)');
ylabel('temperature (°C)');