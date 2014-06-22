package repport;

import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.LinkedHashMap;
import java.util.LinkedList;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ConverterUtils.DataSource;

import common.IOR;
import common.Common;
import common.Progress;

public class section4G {
	// if the following is equal to true, results will be stored in a new folder in the result directory
	// which name will be the date. Otherwise, the folder will have an explicit name (like section21_oneHourAhead for exemple)
	static boolean new_folder_for_results=false;
	
	/**
	 * This file allows us to reproduce simply the results from the section 4G
	 *  Like every other file, it assumes that the data is located in a folder called "aggregated_data" located at 
	 *  root of working directory that contains as many file as there are houses, organized as [timestamp,consum;timestamp,consum...]
	 *  It is exclusively dedicated for the 24 hours ahead problem
	 * @throws Exception 
	 *  
	 */
	public static void main(String[] args) throws Exception{
		buildDatasets();
		predict();
	}

	
	public static void buildDatasets() throws IOException{

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
		header+="@ATTRIBUTE consum_time_d_4 NUMERIC\n";
		header+="@ATTRIBUTE consum_time_d_5 NUMERIC\n";
		header+="@ATTRIBUTE consum_time_d_6 NUMERIC\n";
		header+="@ATTRIBUTE consum_day_7 NUMERIC\n";
		header+="@ATTRIBUTE temperature NUMERIC\n";
		
		
		
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
		
		// load the temperature file, it cannot hurts
		double[] temps = null;
		LinkedList<String> ls_temp=IOR.loadFile("temperatures", "temperatures.txt");
		temps=new double[ls_temp.size()];
		int i=0;
		for(String line : ls_temp){
			temps[i]=Double.parseDouble(line);
			i++;
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
				sb.append(","+consum_houses.get(house)[t-4*24]);
				sb.append(","+consum_houses.get(house)[t-5*24]);
				sb.append(","+consum_houses.get(house)[t-6*24]);
				sb.append(","+consum_houses.get(house)[t-7*24]);
				sb.append(","+temps[t]);
				
				arff_file.add(sb.toString());
			}
			
			// save file
			IOR.saveFile(arff_file, "generated_datasets/section4G_24HoursAhead", house+".arff");

		}
	}
	
	public static void predict() throws Exception{
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

		folder_dataset="generated_datasets/section4G_24HoursAhead";
		output_folder="generated_results/section4G_24HoursAhead";

		exploitationDataset(Common.interesting_houses, tab_classifiers, labels_classifiers, folder_dataset, output_folder);
		
	}
	/*
	 * For this problem we can not use the functions of the section 2.1 : 
	 *  even if the dataset features the consumption at time t-1,t-2,t-3, 
	 *  those ones are only used during training, and not during the prediction
	 *  when we use only datas that have been predicted
	 */
	public static void exploitationDataset(String[] interesting_houses, Classifier[] tab_classifiers, String[] labels_classifiers, String folder_dataset, String output_folder) throws Exception{
		
		System.out.println("starting");
		long time_start=System.currentTimeMillis();
		
		if (new_folder_for_results){
			DateFormat dateFormat = new SimpleDateFormat("yyyy_MM_dd_HH_mm_ss");
			Date date = new Date();
			output_folder=dateFormat.format(date);
		}
		

		int number_of_classifier=tab_classifiers.length;

		Progress.initialize(1, interesting_houses.length);
			// launching the computation for every interesting house
		for (String house : interesting_houses){
			Progress.showProgress();
			// first dataset
			// we import the data of the house in question
			
			String path=IOR.getPathRoot()+folder_dataset+"/"+house+".arff";
			DataSource source_house = new DataSource(path);
			Instances data_house=source_house.getDataSet();
			
			// then create datasets and nominal_datasets, for the classifiers that run on discretized values
			Instances[] sets=section4A.createTrainingAndTestSet(data_house);
			
		    
			// establish prediction and put it in a right file
			int nb_predictions=0;
			for (int i=0;i<number_of_classifier;i++){
				double[] prediction=makePrediction(sets[0], sets[1], tab_classifiers[i]);
				String folder=output_folder+"/"+labels_classifiers[i];
				IOR.saveDoubleArray(prediction, folder, house+".txt");
				nb_predictions=prediction.length;
			}
			
			
			// finally build the comparison with the classifier that gives the same result as 24 hours before
			double[] reals=data_house.attributeToDoubleArray(0);
			double[] reals_test=new double[nb_predictions], reals_24h=new double[nb_predictions];
			for (int i=0; i<nb_predictions;i++){
				reals_test[i]=reals[reals.length-nb_predictions+i];
				reals_24h[i]=reals[reals.length-nb_predictions+i-24];
			}
			String folder=output_folder+"/real";
			IOR.saveDoubleArray(reals_test, folder, house+".txt");
			
			folder=output_folder+"/real_24";
			IOR.saveDoubleArray(reals_24h, folder, house+".txt");
			
			sets=section4A.createTrainingAndTestSet(data_house);
			double[] reals_2=sets[1].attributeToDoubleArray(0);
			
			IOR.saveDoubleArray(reals_2, output_folder+"/real_2", house+".txt");
			
		}
		long time_end=System.currentTimeMillis();
		System.out.println("done, in "+(time_end-time_start)+"ms");
	}
	
	/*
	 * This function gives the result of the application of a classifier to evaluate 
	 * the forecasted consumption of one house
	 * It returns the predicted values
	 */

	public static double[] makePrediction(Instances training_set,Instances test_set, Classifier classifier) throws Exception{	
		// build the classifier
		training_set.setClassIndex(0);
		test_set.setClassIndex(0);
		classifier.buildClassifier(training_set);
		System.out.println(test_set.instance(0).value(0));
	
		// evaluate
		 Evaluation eval = new Evaluation(training_set);
		 double[] prediction=new double[test_set.numInstances()];
		 for (int i=0;i<test_set.numInstances();i++){
			 Instance current_instance=test_set.instance(i);
			 if (i<4){
				 prediction[i]=eval.evaluateModelOnce(classifier, current_instance);
			 }
			 else{
					Instance modified_instance=new Instance(current_instance);
					modified_instance.setDataset(test_set);
					modified_instance.setValue(3, prediction[i-1]);
					modified_instance.setValue(4, prediction[i-2]);
					modified_instance.setValue(5, prediction[i-3]);
					prediction[i]=eval.evaluateModelOnce(classifier, modified_instance);
			 }
		 }
		 return prediction;
	}

}
