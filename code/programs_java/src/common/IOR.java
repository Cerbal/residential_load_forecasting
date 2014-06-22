package common;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.LinkedList;

/*
 * This class simplify the reading and writing of files
 */
public class IOR {
	// the string that contains the path_root.
	// will be fill the first time we use it
	private static String path_root=null;
	
	/* this really simple function load a file into a LinkedList of String
	 * it does not take into account the empty ones
	 */

	public static LinkedList<String> loadFile(String folder, String file) throws IOException{
		FileReader fr=new FileReader(getPathRoot()+folder+"/"+file); 
		BufferedReader br=new BufferedReader(fr);
		String ligne;
		LinkedList<String> list= new LinkedList<String>();
		while ((ligne=br.readLine())!=null){
			if (ligne!="") // does not take into account empty line
				list.add(ligne);
		}
		br.close();
		fr.close();
		return list;
	}
	
	/*
	 * simple accessor to the pathRoot
	 * if path_root is not yet defined, then we define it now
	 */
	public static String getPathRoot() throws IOException{
		if (path_root==null){
			path_root=new java.io.File( "../" ).getCanonicalPath();
			path_root+="/";
		}
		return path_root;
	}
	
	/* this really simple function just store a LinkedList of String into a File
	 * 
	 */
	public static void saveFile(LinkedList<String> data, String folder, String file) throws IOException{
		File dir=new File(getPathRoot()+folder);
		dir.mkdir();
		FileWriter fw=new FileWriter(getPathRoot()+folder+"/"+file);
		BufferedWriter bw= new BufferedWriter(fw, 8192);
		for (String s2 : data){
			bw.write(s2);
			bw.newLine();
		}
		bw.flush();
		bw.close();
		fw.close();
	}
	
	public static void saveDoubleArray(double[] values, String folder, String file) throws IOException{
		File dir=new File(getPathRoot()+folder);
		dir.mkdirs();
		FileWriter fw=new FileWriter(getPathRoot()+folder+"/"+file);
		BufferedWriter bw= new BufferedWriter(fw, 8192);
		for (double d : values){
			double d1=Math.round(1000*d)/1000.0;
			bw.write(String.valueOf(d1));
			bw.newLine();
		}
		bw.flush();
		bw.close();
		fw.close();
	}
	
	public static void printlnTab(double[] tab){
		System.out.println();
		for (int i=0;i<tab.length;i++){
			System.out.print(String.valueOf(tab[i])+",");
		}
		System.out.println();
	}
	
	// in fact this has nothing to do in IOR, but I did not know where to put it
	public static double minTab(double[] tab){
		double temp=100000000;
		for (int i=0; i<tab.length; i++){
			if (tab[i]<temp){
				temp=tab[i];
			}
		}
		return temp;
	}
	
	// in fact this has nothing to do in IOR, but I did not know where to put it
	public static double maxTab(double[] tab){
		double temp=-100000000;
		for (int i=0; i<tab.length; i++){
			if (tab[i]>temp){
				temp=tab[i];
			}
		}
		return temp;
	}
	
	/* this function crawls the data
	 * i.e. : it will import the values contained in files called "(num. house).txt"
	 * and put them into array of double.
	 * It returns an hashmap
	 */
	public static LinkedHashMap<String, double[]> crawlHouses(LinkedList<String> index, String folder, int nb_lines, boolean withTimeStamp) throws IOException{
		LinkedHashMap<String,double[]> consum_houses=new LinkedHashMap<String,double[]>();
		Progress.initialize(1, index.size());		
		for(String house : index){
			Progress.showProgress(5);
			LinkedList<String> data_house_f=IOR.loadFile(folder, house+".txt");
			int count=0;
			double[] data_tab=new double[nb_lines];
			for(String data_value : data_house_f){
				if (withTimeStamp)
					data_tab[count]=Double.parseDouble((data_value.split(","))[1]);
				else
					data_tab[count]=Double.parseDouble(data_value);
				count++;
			}
			consum_houses.put(house, data_tab);
		}
		return consum_houses;
	}
}
