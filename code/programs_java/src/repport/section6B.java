package repport;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.LinkedList;

import weka.classifiers.Classifier;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.SelectedTag;

import common.IOR;
import common.Common;
import common.Progress;

public class section6B {
	// if the following is equal to true, results will be stored in a new folder in the result directory
	// which name will be the date. Otherwise, the folder will have an explicit name (like section21_oneHourAhead for exemple)
	static boolean new_folder_for_results=false;
	

	/**
	 * This file allows us to reproduce simply the results from the section 6B
	 *  Like every other file, it assumes that the data is located in a folder called "aggregated_data" located at 
	 *  root of working directory that contains as many file as there are houses, organized as [timestamp,consum;timestamp,consum...]
	 * @throws Exception 
	 *  
	 */
	public static void main(String[] args) throws Exception {
		buildDatasets(false);
		buildDatasets(true);
		predict(true);
		predict(false);
	}
	
	
	/*
	 * This function build the simple datasets used in section 2.1
	 */
	public static void buildDatasets(boolean leadersCrossCorrelation) throws IOException{

		// at first the header
		String header="@ATTRIBUTE consum_time_t NUMERIC\n";
		header+="@ATTRIBUTE day_week NUMERIC\n";
		header+="@ATTRIBUTE hour_day NUMERIC\n";

		header+="@ATTRIBUTE consum_time_t_1 NUMERIC\n";
		header+="@ATTRIBUTE consum_time_t_2 NUMERIC\n";
		header+="@ATTRIBUTE consum_time_t_3 NUMERIC\n";

		header+="@ATTRIBUTE consum_time_d_1 NUMERIC\n";
		header+="@ATTRIBUTE consum_time_d_2 NUMERIC\n";
		header+="@ATTRIBUTE consum_time_d_3 NUMERIC\n";
		header+="@ATTRIBUTE consum_day_7 NUMERIC\n";
		
		int nb_leaders=4; // the number of considered leaders
		for (int i=0;i<nb_leaders;i++){
			header+="@ATTRIBUTE leader"+i+" NUMERIC\n";
		}
		
		//go look for the index that will give the leaders and the lag for each house
		// by convention the house himself at time t-1 figure among the leaders in first position
		// leaders have been generated in Matlab
		LinkedHashMap<String, String[]> leaders_id=new LinkedHashMap<String, String[]>();
		LinkedHashMap<String, int[]> leaders_lag=new LinkedHashMap<String, int[]>();
		LinkedList<String> leaders_f=null;
		if (leadersCrossCorrelation)
			leaders_f= IOR.loadFile("aggregated_data", "leaders.txt");
		else
			leaders_f= IOR.loadFile("aggregated_data", "leaders_8bins.txt");
		
		for (String line : leaders_f){
			String[] lines=line.split(",");
			String[] tab_id;
			int[] tab_lag;
			tab_id=new String[nb_leaders];
			tab_lag=new int[nb_leaders];

			String house=String.valueOf(((int) Double.parseDouble(lines[0])));
			for (int i=0;i<nb_leaders;i++){
				tab_id[i]=String.valueOf(((int) Double.parseDouble(lines[(i+1)*3])));
				tab_lag[i]=(int) Double.parseDouble(lines[(i+1)*3+1]);
			}
			// the main leader of the time series is itself at time t-1
			leaders_id.put(house, tab_id);
			leaders_lag.put(house, tab_lag);
		}
		
		
		
		// get the ids of the houses     
		System.out.println("starting");
		LinkedList<String> index_houses=IOR.loadFile("aggregated_data", "index.txt");
		
		//load one file, just to obtain the number of line, and the time stamp
		LinkedList<String> file=IOR.loadFile("aggregated_data", "1002.txt");
		int nb_lines=file.size();
		int[] timestamps=new int[nb_lines];
		int count=0;
		for (String line : file){
			timestamps[count]=Integer.parseInt(line.split(",")[0]);
			count++;
		}
		
		// at first, we store the entire data set for the houses in one array
		System.out.println("acquiring the whole dataset");
		// create the dictionary that will contain the entire set of data
		LinkedHashMap<String,double[]> consum_houses=IOR.crawlHouses(index_houses, "aggregated_data", nb_lines, true);

	
		System.out.println("creating arff files");
		// finally creating the arff file
		Progress.initialize(1, index_houses.size());		
		for(String house : index_houses){
			Progress.showProgress(5);
			LinkedList<String> arff_file= new LinkedList<String>();
			arff_file.add("@RELATION house_"+house+"\n\n");
			arff_file.add(header+"\n\n @DATA");
			for (int t=7*24+1;t<nb_lines;t++){
				StringBuilder sb=new StringBuilder();
				
				int timestamp=timestamps[t];
				int[] ts=Common.transformTimeStamp(timestamp);
				
				sb.append(consum_houses.get(house)[t]);
				sb.append(","+ts[1]);
				sb.append(","+ts[2]);
				sb.append(","+consum_houses.get(house)[t-1]);
				sb.append(","+consum_houses.get(house)[t-2]);
				sb.append(","+consum_houses.get(house)[t-3]);

				sb.append(","+consum_houses.get(house)[t-1*24]);
				sb.append(","+consum_houses.get(house)[t-2*24]);
				sb.append(","+consum_houses.get(house)[t-3*24]);
				sb.append(","+consum_houses.get(house)[t-7*24]);
				
				// there we put the value of the leaders
				for (int i=0;i<nb_leaders;i++){
					String leader=leaders_id.get(house)[i];
					int lag=leaders_lag.get(house)[i];
					sb.append(","+consum_houses.get(leader)[t-lag]);
				}
				
				arff_file.add(sb.toString());
			}
			
			// save file
			if (leadersCrossCorrelation)
				IOR.saveFile(arff_file, "generated_datasets/section6B_leaderCrossCorrelation", house+".arff");
			else
				IOR.saveFile(arff_file, "generated_datasets/section6B_leaderMutualInformation", house+".arff");

		}
	}
	
	/*
	 * prepare the classifiers and launch the prediction
	 */
	public static void predict(boolean leadersCrossCorrelation) throws Exception{
		LinearRegression linreg=new LinearRegression();
		
		// a multi-linear perceptron with 30 of validation set for early stopping (avoid overfitting)
		MultilayerPerceptron mlp=new MultilayerPerceptron();
		mlp.setValidationSetSize(30);
		mlp.setValidationThreshold(20);
		
		// and a SVR, with the optimum gamma and C found in the file VM classifier
		LibSVM svm= new LibSVM();
		svm.setSVMType(new SelectedTag(LibSVM.SVMTYPE_EPSILON_SVR, LibSVM.TAGS_SVMTYPE));
		svm.setGamma(0.01);
		svm.setCost(100);
		svm.setNormalize(true);
		svm.setEps(0.01);
		svm.setDebug(false);
		
		Classifier tab_classifiers[]=new Classifier[]{linreg, svm, mlp};
		String[] labels_classifiers=new String[]{"LinearRegression","SVR","MLP"};
		String folder_dataset;
		String output_folder;
		if (leadersCrossCorrelation){
			folder_dataset="generated_datasets/section6B_leaderCrossCorrelation";
			output_folder="generated_results/section6B_leaderCrossCorrelation";
		}
		else{
			folder_dataset="generated_datasets/section6B_leaderMutualInformation";
			output_folder="generated_results/section6B_leaderMutualInformation";
			
		}
		
		
		section4A.exploitationDataset(Common.interesting_houses, tab_classifiers, labels_classifiers, folder_dataset, output_folder);
		
	}
}
