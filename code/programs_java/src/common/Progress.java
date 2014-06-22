package common;

public class Progress {
	public static int previous=0;
	public static int current=0;
	public static int start=0;
	public static int ending=0;
	public static int stp=2;
	
	public static void showProgress(){
		current++;
		if (previous>=100)
			previous=0;
		if (ending-start==0)
			ending++;
		if (Math.round(100*(current-start)/(ending-start)) >= previous+stp){
			previous+=stp;
			System.out.println(previous+"%");
		}
	}
	
	public static void showProgress(int step){
		stp=step;
		showProgress();
	}
	public static void initialize(int start_point, int end_point){
		start=start_point;
		current=start_point-1;
		ending=end_point;
	}
}
