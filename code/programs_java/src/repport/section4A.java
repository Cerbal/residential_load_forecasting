package repport;

import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.LinkedHashMap;
import java.util.LinkedList;

import common.IOR;
import common.Progress;
import common.Common;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemoveRange;

public class section4A {
	// if the following is equal to true, results will be stored in a new folder in the result directory
	// which name will be the date. Otherwise, the folder will have an explicit name (like section4A_oneHourAhead for exemple)
	static boolean new_folder_for_results=false;
	

	/**
	 * This file allows us to reproduce simply the results from the section 4A
	 *  Like every other file, it assumes that the data is located in a folder called "aggregated_data" located at 
	 *  root of working directory that contains as many file as there are houses, organized as [timestamp,consum;timestamp,consum...]
	 *  It produces datasets in a 
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
	 * This function build the simple datasets used in section 2.1
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
				
				arff_file.add(sb.toString());
			}
			
			// save file
			if (OneHourAhead)
				IOR.saveFile(arff_file, "generated_datasets/section4A_oneHourAhead", house+".arff");
			else
				IOR.saveFile(arff_file, "generated_datasets/section4A_24HoursAhead", house+".arff");

		}
	}
	
	/*
	 * prepare the classifiers and launch the prediction
	 */
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
			folder_dataset="generated_datasets/section4A_oneHourAhead";
			output_folder="generated_results/section4A_oneHourAhead";
		}
		else{
			folder_dataset="generated_datasets/section4A_24HoursAhead";
			output_folder="generated_results/section4A_24HoursAhead";
		}
		
		
		
		exploitationDataset(Common.interesting_houses, tab_classifiers, labels_classifiers, folder_dataset, output_folder);
		
	}
	
	/*
	 * Exploit the dataset using continuous classifiers
	 * (note that during the development phase, I had done a nice function that can deal either with nominal
	 * or continuous classifier.) However it no more useful since in the report I deal separately with the two cases.
	 * As always it stores the result 
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
			Instances[] sets=createTrainingAndTestSet(data_house);
			
		    
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
			double[] reals_test=new double[nb_predictions], reals_24h=new double[nb_predictions], reals_1h=new double[nb_predictions];
			for (int i=0; i<nb_predictions;i++){
				reals_test[i]=reals[reals.length-nb_predictions+i];
				reals_24h[i]=reals[reals.length-nb_predictions+i-24];
				reals_1h[i]=reals[reals.length-nb_predictions+i-1];
			}
			String folder=output_folder+"/real";
			IOR.saveDoubleArray(reals_test, folder, house+".txt");
			
			folder=output_folder+"/real_24";
			IOR.saveDoubleArray(reals_24h, folder, house+".txt");
			
			folder=output_folder+"/real_1";
			IOR.saveDoubleArray(reals_1h, folder, house+".txt");
			
			sets=createTrainingAndTestSet(data_house);
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
	
		// evaluate
		 Evaluation eval = new Evaluation(training_set);
		 double[] predict= eval.evaluateModel(classifier, test_set);	 
		 return predict;
	}

	public static Instances[] createTrainingAndTestSet(Instances instances) throws Exception{
		// split the data set in two parts, training and testing
		// the number of instance in the test set will be 4317
		System.out.println(instances.toSummaryString());
		
		int cesure=instances.numInstances()-4317;
		RemoveRange remove_test=new RemoveRange();
		remove_test.setInputFormat(instances);
		remove_test.setInstancesIndices(String.valueOf(cesure+1)+"-"+String.valueOf(instances.numInstances()));
		Instances data_classification_day=Filter.useFilter(instances, remove_test);
		System.out.println(data_classification_day.numInstances());
		
		RemoveRange remove_train=new RemoveRange();
		remove_train.setInputFormat(instances);
		remove_train.setInstancesIndices("1-"+String.valueOf(cesure));
		Instances data_test_day=Filter.useFilter(instances, remove_train);
		System.out.println(data_test_day.numInstances());
		
		return new Instances[]{data_classification_day,data_test_day};
	}

}
