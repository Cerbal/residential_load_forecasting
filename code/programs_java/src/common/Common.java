package common;

/*
 * A class that regroup various useful functions
 */
public class Common {
	public static String[] interesting_houses=new String[]{"1002","1014","1018","1022","1440","1695","4332", "1843","6568","1859","1807", "2103","2387","4879","2265","2945","3355","4076","4755","3816","1331","6445","5291","3660","1969"};
	//public static String[] interesting_houses=new String[]{"1002","1014"};
	
	/*
	 * This function convert a time stamp into an array of integer
	 * first one : refers to the day in the year
	 * second one : refers to the day of the week
	 * third one : refers to the hour of the day
	 */
	public static int[] transformTimeStamp(int timestamp){
		int[] result=new int[3];
		result[2]=timestamp%100;
		// the first day was a thursday, january first
		int day=(timestamp/100);
		result[0]=day%365;
		result[1]=(day+2)%7+1;
		return result;
	}

}
