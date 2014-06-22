package repport;

import common.*;

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
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemoveRange;

public class section4F {
	// if the following is equal to true, results will be stored in a new folder in the result directory
	// which name will be the date. Otherwise, the folder will have an explicit name (like section21_oneHourAhead for exemple)
	static boolean new_folder_for_results=false;
	
	/**
	 * This file allows us to reproduce simply the results from the section 4F
	 *  Like every other file, it assumes that the data is located in a folder called "aggregated_data" located at 
	 *  root of working directory that contains as many file as there are houses, organized as [timestamp,consum;timestamp,consum...]
	 *  It produces two datasets, each in their corresponding folder
	 *  It also assumes that baselines are stored in folder baselines/'name of baseline'
	 * @throws Exception 
	 *  
	 */
	public static void main(String[] args) throws Exception {
		buildDatasets();
		predict();
	}
	
	/*
	 * This function create two simple datasets (one for one hour ahead prediction and one for 24 hours ahead prediction)
	 * that include previous consumption and baselines.
	 *  
	 */
	public static void buildDatasets() throws IOException{
		String header="@ATTRIBUTE consum_time_t NUMERIC\n";
		header+="@ATTRIBUTE day_week NUMERIC\n";
		header+="@ATTRIBUTE hour_day NUMERIC\n";

		header+="@ATTRIBUTE consum_time_t_1 NUMERIC\n";
		header+="@ATTRIBUTE consum_time_t_2 NUMERIC\n";
		header+="@ATTRIBUTE consum_time_t_3 NUMERIC\n";
		
		header+="@ATTRIBUTE gradient_time_t_1 NUMERIC\n";
		header+="@ATTRIBUTE gradient_time_t_2 NUMERIC\n";
		header+="@ATTRIBUTE gradient_time_t_3 NUMERIC\n";
		
		header+="@ATTRIBUTE lagrangian_time_t_1 NUMERIC\n";
		header+="@ATTRIBUTE lagrangian_time_t_2 NUMERIC\n";
		header+="@ATTRIBUTE lagrangian_time_t_3 NUMERIC\n";
		
		header+="@ATTRIBUTE consum_time_d_1 NUMERIC\n";
		header+="@ATTRIBUTE consum_time_d_2 NUMERIC\n";
		header+="@ATTRIBUTE consum_time_d_3 NUMERIC\n";
		header+="@ATTRIBUTE consum_day_7 NUMERIC\n";
		
		header+="@ATTRIBUTE temperature NUMERIC\n";
		
		header+="@ATTRIBUTE baselineISONE NUMERIC\n";
		header+="@ATTRIBUTE baselineMid4of6 NUMERIC\n";
		header+="@ATTRIBUTE baselinePMJEco NUMERIC\n";
		


		
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
		

		// load the temperature file, it cannot hurts
		double[] temps = null;
		LinkedList<String> ls_temp=IOR.loadFile("temperatures", "temperatures.txt");
		temps=new double[ls_temp.size()];
		int i=0;
		for(String line : ls_temp){
			temps[i]=Double.parseDouble(line);
			i++;
		}
		
		
		System.out.println("creating arff files");
		// finally creating the arff file
		// one file for day and night
		
		Progress.initialize(1, index_houses.size());		
		for(String house : index_houses){
			Progress.showProgress(5);
			LinkedList<String> arff_file_day= new LinkedList<String>();
			LinkedList<String> arff_file_night= new LinkedList<String>();
			arff_file_day.add("@RELATION house_"+house+"\n\n");
			arff_file_day.add(header+"\n\n @DATA");
			
			arff_file_night.add("@RELATION house_"+house+"\n\n");
			arff_file_night.add(header+"\n\n @DATA");
			
			double[] tab_day=new double[nb_lines-1-7*24];
			for (int t=7*24+1;t<nb_lines;t++){
				StringBuilder sb=new StringBuilder();
				
				int timestamp=timestamps[t];
				int[] ts=Common.transformTimeStamp(timestamp);
				boolean day=true;
				if (ts[2]>=1 && ts[2]<=7)
					day=false;
				
				sb.append(consum_houses.get(house)[t]);
				sb.append(","+ts[1]);
				sb.append(","+ts[2]);
				
				// consumptions
				sb.append(","+consum_houses.get(house)[t-1]);
				sb.append(","+consum_houses.get(house)[t-2]);
				sb.append(","+consum_houses.get(house)[t-3]);
				
				// gradients
				sb.append(","+(consum_houses.get(house)[t-1]-consum_houses.get(house)[t-2]));
				sb.append(","+(consum_houses.get(house)[t-2]-consum_houses.get(house)[t-3]));
				sb.append(","+(consum_houses.get(house)[t-3]-consum_houses.get(house)[t-4]));
				
				// Lagrangian
				sb.append(","+(consum_houses.get(house)[t-1]+consum_houses.get(house)[t-3]-2*consum_houses.get(house)[t-2]));
				sb.append(","+(consum_houses.get(house)[t-2]+consum_houses.get(house)[t-4]-2*consum_houses.get(house)[t-3]));
				sb.append(","+(consum_houses.get(house)[t-3]+consum_houses.get(house)[t-5]-2*consum_houses.get(house)[t-4]));

				sb.append(","+consum_houses.get(house)[t-1*24]);
				sb.append(","+consum_houses.get(house)[t-2*24]);
				sb.append(","+consum_houses.get(house)[t-3*24]);

				sb.append(","+consum_houses.get(house)[t-7*24]);
				sb.append(","+temps[t]);
				sb.append(","+isone.get(house)[t]);
				sb.append(","+mid4of6.get(house)[t]);
				sb.append(","+pjmEco.get(house)[t]);
				
				if (day){
					tab_day[t-24*7-1]=1;
					arff_file_day.add(sb.toString());
				}
				else{
					tab_day[t-24*7-1]=0;
					arff_file_night.add(sb.toString());
					}
				
				
			}

			IOR.saveFile(arff_file_day, "generated_datasets/section4F", house+"_Day.arff");
			IOR.saveFile(arff_file_night, "generated_datasets/section4F", house+"_Night.arff");
			IOR.saveDoubleArray(tab_day,"generated_datasets/section4F", house+"_index.txt");
		}
	
	}

	public static void predict() throws Exception{
		
		LinearRegression linreg=new LinearRegression();
		//a multi-linear perceptron with 30 of validation set for early stopping (avoid overfitting)
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
		
		// and we redo it for the night
		// notice that this repetition allows to choose different classifier for night and day
		LinearRegression linreg2=new LinearRegression();
		//a multi-linear perceptron with 30 of validation set for early stopping (avoid overfitting)
		MultilayerPerceptron mlp2=new MultilayerPerceptron();
		mlp2.setValidationSetSize(30);
		mlp2.setValidationThreshold(20);
		// and a SVR, with the optimum gamma and C found in the file VM classifier
		LibSVM svm2= new LibSVM();
		svm2.setSVMType(new SelectedTag(LibSVM.SVMTYPE_EPSILON_SVR, LibSVM.TAGS_SVMTYPE));
		svm2.setGamma(0.01);
		svm2.setCost(100);
		svm2.setNormalize(true);
		svm2.setEps(0.01);
		svm2.setDebug(false);
		
		Classifier tab_classifiers_day[]=new Classifier[]{linreg, svm, mlp};
		Classifier tab_classifiers_night[]=new Classifier[]{linreg2, svm2, mlp2};
		String[] labels_classifiers=new String[]{"LinearRegression","SVR","MLP"};
		String folder_dataset;
		
		folder_dataset="generated_datasets/section4F";
		String output_dataset="generated_results/section4F_oneHourAhead";

		
		exploitationDatasetNightDay(Common.interesting_houses, tab_classifiers_day, tab_classifiers_night, labels_classifiers, folder_dataset, output_dataset);
	}

	public static void exploitationDatasetNightDay(String[] interesting_houses, Classifier[] tab_classifiers_day,  Classifier[] tab_classifiers_night, String[] labels_classifiers, String folder_dataset, String output_folder) throws Exception{
		
		System.out.println("starting");
		long time_start=System.currentTimeMillis();
		
		if (new_folder_for_results){
			DateFormat dateFormat = new SimpleDateFormat("yyyy_MM_dd_HH_mm_ss");
			Date date = new Date();
			output_folder=dateFormat.format(date);
		}
		
		

		int number_of_classifier=tab_classifiers_day.length;

		Progress.initialize(1, interesting_houses.length);
			// launching the computation for every interesting house
		for (String house : interesting_houses){
			Progress.showProgress();

			// we import the data of the house in question
			DataSource source_house_day = new DataSource(IOR.getPathRoot()+folder_dataset+"/"+house+"_Day.arff");
			Instances data_house_day=source_house_day.getDataSet();
			DataSource source_house_night = new DataSource(IOR.getPathRoot()+folder_dataset+"/"+house+"_Night.arff");
			Instances data_house_night=source_house_night.getDataSet();
			LinkedList<String> index_day_nightf= IOR.loadFile(folder_dataset, house+"_index.txt");
			int[] index_day_night=new int[index_day_nightf.size()];
			int count=0;
			for(String line : index_day_nightf){
				index_day_night[count]=(int) Double.parseDouble(line);
				count++;
			}	
			
			// then create datasets and nominal_datasets, for the classifiers that run on discretized values
			Instances[] sets=createTrainingAndTestSetNightDay(data_house_day, data_house_night, index_day_night);
			
		    
			// establish prediction and put it in a right file
			for (int i=0;i<number_of_classifier;i++){
				double[] prediction=makePredictionNightDay(sets, tab_classifiers_day[i], tab_classifiers_night[i],index_day_night);
				String folder=output_folder+"/"+labels_classifiers[i];
				IOR.saveDoubleArray(prediction, folder, house+".txt");
			}
			
			
		}
		long time_end=System.currentTimeMillis();
		System.out.println("done, in "+(time_end-time_start)+"ms");
	}
	
	/*
	 * This function gives the result of the application of a classifier to evaluate 
	 * the forecasted consumption of one house
	 * It returns the predicted values
	 */

	public static double[] makePredictionNightDay(Instances[] sets, Classifier classifier_day, Classifier classifier_night,int[] index_day_night) throws Exception{	
		Instances train_day=sets[0];
		Instances test_day=sets[1];
		Instances train_night=sets[2];
		Instances test_night=sets[3];
		
		train_day.setClassIndex(0);
		test_day.setClassIndex(0);
		classifier_day.buildClassifier(train_day);
		
		train_night.setClassIndex(0);
		test_night.setClassIndex(0);
		classifier_night.buildClassifier(train_night);
		
		// evaluations
		 Evaluation eval_day = new Evaluation(train_day);
		 Evaluation eval_night = new Evaluation(train_night);
		 eval_day.evaluateModel(classifier_day, test_day);
		 int nb_prediction=test_night.numInstances()+test_day.numInstances();
		
		 double[] predictions=new double[nb_prediction];
		 
		// now that we have a prediction for the day and for the night, we merge them
		int count_day=0,count_night=0;
		int count=0;
		for (int t=index_day_night.length-nb_prediction;t<index_day_night.length;t++){
			Instance current_instance;
			double predict_once=0;
			if (index_day_night[t]==1){
				current_instance=test_day.instance(count_day);
				predict_once=eval_day.evaluateModelOnce(classifier_day, current_instance);
				count_day++;
			}
			else{
				current_instance=test_night.instance(count_night);
				predict_once=eval_night.evaluateModelOnce(classifier_night, current_instance);
				count_night++;
			}
			predictions[count]=predict_once;
			count++;
		}
		
		return predictions;
	}

	public static Instances[] createTrainingAndTestSetNightDay(Instances instances_day, Instances instances_night, int[] index_daynight) throws Exception{
		// the goal is to make a 67-33. 
		
		// the difficulty here is that the dataset is splitted into two parts
		// we must have something coherent at the end
		// at first we determine how many day and night tupple in each part of the dataset
		
		int cesure=index_daynight.length-4317;
		int count_day=0,count_night=0;
		int count_day_all=0,count_night_all=0;
		for(int count=0;count<index_daynight.length; count++){
			int day=index_daynight[count];
			if (day==1){
				if(count<cesure)
					count_day++;
				count_day_all++;
			}
			else{
				if (count<cesure)
					count_night++;
				count_night_all++;
			}
		}
		
		// split the data set in two parts, training and testing
		RemoveRange remove_test=new RemoveRange();
		remove_test.setInputFormat(instances_day);
		remove_test.setInstancesIndices(String.valueOf(count_day+1)+"-"+String.valueOf(count_day_all));
		Instances data_classification_day=Filter.useFilter(instances_day, remove_test);
		remove_test=new RemoveRange();
		remove_test.setInputFormat(instances_day);
		remove_test.setInstancesIndices("1-"+String.valueOf(count_day));
		Instances data_test_day=Filter.useFilter(instances_day, remove_test);
		
		remove_test=new RemoveRange();
		remove_test.setInputFormat(instances_night);
		remove_test.setInstancesIndices(String.valueOf(count_night+1)+"-"+String.valueOf(count_night_all));
		Instances data_classification_night=Filter.useFilter(instances_night, remove_test);
		remove_test=new RemoveRange();
		remove_test.setInputFormat(instances_night);
		remove_test.setInstancesIndices("1-"+String.valueOf(count_night));
		Instances data_test_night=Filter.useFilter(instances_night, remove_test);

		return new Instances[]{data_classification_day, data_test_day, data_classification_night, data_test_night};
	}


}
