function dates=convertTimestamp(timestamp)
        result=[2010,0,0,0,0,0];
        days={'mon.','tue.','wed.','thu.','fri.','sat.','sun.'};
		day=floor(timestamp/100)-365;
        hour=mod(timestamp,100);
        result(3)=day;
        result(4)=hour;
		week_day=mod(day+3,7)+1;
        temp=strcat(days(week_day),datestr(result,':HH-MM-SS-yyyy-mm-dd'));
        dates=temp{1};
end