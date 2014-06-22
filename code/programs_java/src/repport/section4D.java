package repport;

import common.*;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.LinkedList;

import weka.classifiers.Classifier;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.SelectedTag;

public class section4D {
	/**
	 * This file allows us to reproduce simply the results from the section 4D
	 *  Like every other file, it assumes that the data is located in a folder called "aggregated_data" located at 
	 *  root of working directory that contains as many file as there are houses, organized as [timestamp,consum;timestamp,consum...]
	 *  It produces two datasets, each in their corresponding folder
	 *  It also assumes that baselines are 
	 * @throws Exception 
	 *  
	 */
	public static void main(String[] args) throws Exception {
		buildDatasets(false);
		predict(false);
		buildDatasets(true);
		predict(true);
	}
	
	/*
	 * This function create two simple datasets (one for one hour ahead prediction and one for 24 hours ahead prediction)
	 * that include previous consumption and baselines.
	 *  
	 */
	public static void buildDatasets(boolean OneHourAhead) throws IOException{

		// at first the header
		String header="@ATTRIBUTE consum_time_t NUMERIC\n";
		header+="@ATTRIBUTE day_week NUMERIC\n";
		header+="@ATTRIBUTE hour_day NUMERIC\n";
		if (OneHourAhead){
			header+="@ATTRIBUTE consum_time_t_1 NUMERIC\n";
			header+="@ATTRIBUTE consum_time_t_2 NUMERIC\n";
			header+="@ATTRIBUTE consum_time_t_3 NUMERIC\n";
		}
		header+="@ATTRIBUTE consum_time_d_1 NUMERIC\n";
		header+="@ATTRIBUTE consum_time_d_2 NUMERIC\n";
		header+="@ATTRIBUTE consum_time_d_3 NUMERIC\n";
		if (!OneHourAhead){
			header+="@ATTRIBUTE consum_time_d_4 NUMERIC\n";
			header+="@ATTRIBUTE consum_time_d_5 NUMERIC\n";
			header+="@ATTRIBUTE consum_time_d_6 NUMERIC\n";
		}
		header+="@ATTRIBUTE consum_day_7 NUMERIC\n";
		header+="@ATTRIBUTE temperature NUMERIC\n";
		header+="@ATTRIBUTE baselineISONE NUMERIC\n";
		header+="@ATTRIBUTE baselineMid4of6 NUMERIC\n";
		header+="@ATTRIBUTE baselinePMJEco NUMERIC\n";
		
		
		// load the temperature file, it cannot hurts
		double[] temps = null;
		LinkedList<String> ls_temp=IOR.loadFile("temperatures", "temperatures.txt");
		temps=new double[ls_temp.size()];
		int i=0;
		for(String line : ls_temp){
			temps[i]=Double.parseDouble(line);
			i++;
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
		
		// then we import (once for all) the baselines (it may takes a bit of time)
		System.out.println("acquire ISOONE");
		LinkedHashMap<String,double[]> isone = IOR.crawlHouses(index_houses, "baselines/ISOONE", nb_lines, false);
		System.out.println("acquire middle4of6");
		LinkedHashMap<String,double[]> mid4of6 = IOR.crawlHouses(index_houses, "baselines/middle4of6", nb_lines, false);
		System.out.println("acquire PJMEconomic");
		LinkedHashMap<String,double[]> pjmEco = IOR.crawlHouses(index_houses, "baselines/PJMEconomic", nb_lines, false);

	
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
				if (OneHourAhead){
					sb.append(","+consum_houses.get(house)[t-1]);
					sb.append(","+consum_houses.get(house)[t-2]);
					sb.append(","+consum_houses.get(house)[t-3]);
				}
				sb.append(","+consum_houses.get(house)[t-1*24]);
				sb.append(","+consum_houses.get(house)[t-2*24]);
				sb.append(","+consum_houses.get(house)[t-3*24]);
				if (!OneHourAhead){
					sb.append(","+consum_houses.get(house)[t-4*24]);
					sb.append(","+consum_houses.get(house)[t-5*24]);
					sb.append(","+consum_houses.get(house)[t-6*24]);
				}
				sb.append(","+consum_houses.get(house)[t-7*24]);
				sb.append(","+temps[t]);
				sb.append(","+isone.get(house)[t]);
				sb.append(","+mid4of6.get(house)[t]);
				sb.append(","+pjmEco.get(house)[t]);
				
				arff_file.add(sb.toString());
			}
			
			// save file
			if (OneHourAhead)
				IOR.saveFile(arff_file, "generated_datasets/section4D_oneHourAhead", house+".arff");
			else
				IOR.saveFile(arff_file, "generated_datasets/section4D_24HoursAhead", house+".arff");

		}
	}

	public static void predict(boolean OneHourAhead) throws Exception{
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
		if (OneHourAhead){
			folder_dataset="generated_datasets/section4D_oneHourAhead";
			output_folder="generated_results/section4D_oneHourAhead";
		}
		else{
			folder_dataset="generated_datasets/section4D_24HoursAhead";
			output_folder="generated_results/section4D_24HoursAhead";
		}
		
		// we use directly the function that have been written in section 21.java
		section4A.exploitationDataset(Common.interesting_houses, tab_classifiers, labels_classifiers, folder_dataset, output_folder);
	}
}
