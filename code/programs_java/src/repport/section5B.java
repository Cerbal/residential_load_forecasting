package repport;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;

import common.CTTMFilter;
import common.Common;
import common.IOR;
import common.Progress;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

public class section5B {
	// if the following is equal to true, results will be stored in a new folder in the result directory
	// which name will be the date. Otherwise, the folder will have an explicit name (like section21_oneHourAhead for exemple)
	static boolean new_folder_for_results=false;
	
	/**
	 * This file allows us to reproduce simply the results from the section 5B
	 * Notice that this one does not create any dataset. Since it simply discretize the dataset from section 4B
	 * @throws Exception 
	 *  
	 */
	public static void main(String[] args) throws Exception{
		predict(false);
		predict(true);
	}
	/*
	 * Discretize and predict
	 */
	public static void predict(boolean useCTTM) throws Exception{
		
		// a multi-linear perceptron with 30 of validation set for early stopping (avoid overfitting)
		J48 j48=new J48();
		NaiveBayes nb=new NaiveBayes();
		RandomForest rf=new RandomForest();
		
		// and a SVR, with the optimum gamma and C found in the file VM classifier
		LibSVM svm= new LibSVM();
		svm.setSVMType(new SelectedTag(LibSVM.SVMTYPE_C_SVC, LibSVM.TAGS_SVMTYPE));
		svm.setGamma(0.01);
		svm.setCost(100);
		svm.setNormalize(true);
		svm.setEps(0.01);
		svm.setDebug(false);
		
		Classifier tab_classifiers[]=new Classifier[]{j48, nb, rf, svm};
		String[] labels_classifiers=new String[]{"J48","NaiveBayes","RandomForest", "SVM"};
		
		String folder_dataset;
		folder_dataset="generated_datasets/section4A_oneHourAhead";
		String output_folder;
		if (useCTTM)
			output_folder="generated_results/section5B_cttm";
		else
			output_folder="generated_results/section5B_unif";

		exploitationDatasetNominal(Common.interesting_houses, useCTTM, tab_classifiers, labels_classifiers, folder_dataset, output_folder);
	}
	
	/*
	 * Exploit the dataset using nominal classifiers
	 * (note that during the development phase, I had done a nice function that can deal either with nominal
	 * or continuous classifier.) However it no more useful since in the report I deal separately with the two cases.
	 */
	public static void exploitationDatasetNominal(String[] interesting_houses, boolean useCTTM, Classifier[] tab_classifiers, String[] labels_classifiers, String folder_dataset, String output_folder) throws Exception{
		
		System.out.println("starting");
		long time_start=System.currentTimeMillis();
		
		// create a string that contains launch date, if needed
		if (new_folder_for_results){
			DateFormat dateFormat = new SimpleDateFormat("yyyy_MM_dd_HH_mm_ss");
			Date date = new Date();
			output_folder=dateFormat.format(date);
		}
		int number_of_classifier=tab_classifiers.length;

		// launching the computation for every interesting house
		Progress.initialize(1, interesting_houses.length);
		for (String house : interesting_houses){
			Progress.showProgress();

			
			// we import the data of the house in question
			String path=IOR.getPathRoot()+folder_dataset+"/"+house+".arff";
			DataSource source_house = new DataSource(path);
			Instances data_house=source_house.getDataSet();
			
			// then create training and test set
			Instances[] continous_sets=section4A.createTrainingAndTestSet(data_house);
			Instances[] sets=new Instances[2];
			
		    // one the nominal dataset is created we need to establish what values will be outputed
		    double[] slices=null;
			
			// Normalize on 8 bits
			if (useCTTM){
				CTTMFilter dis_filter= new CTTMFilter(8, "0,3-9");
				dis_filter.establishBounds(continous_sets[0]);
				sets[0]=dis_filter.process(continous_sets[0]);
				sets[1]=dis_filter.process(continous_sets[1]);
				slices=dis_filter.val_interval[0];
			}
			else{
				// otherwise we use a normal uniform discretization
				Discretize dis_filter = new Discretize();
				dis_filter.setAttributeIndices("1,4-10");
			    dis_filter.setInputFormat(continous_sets[0]);
			    dis_filter.setBins(8);
			    dis_filter.setUseEqualFrequency(false);
			    sets[0]=Filter.useFilter(continous_sets[0], dis_filter);
			    sets[1]=Filter.useFilter(continous_sets[1], dis_filter);
			    double[] cut_points=dis_filter.getCutPoints(0);
			    double[] values_consum=sets[0].attributeToDoubleArray(0);
			    double min_value=IOR.minTab(values_consum);
			    double max_value=IOR.maxTab(values_consum);
			    slices=new double[cut_points.length+1];
			    slices[0]=(min_value+cut_points[0])/2;
			    slices[cut_points.length]=(max_value+cut_points[cut_points.length-1])/2;
			    for (int i=1;i<cut_points.length;i++){
			    	slices[i]=(cut_points[i]+cut_points[i-1])/2;
			    }
			}
			
			
		    
			// establish prediction and put it in a right file
			int nb_predictions=0;
			for (int i=0;i<number_of_classifier;i++){
				double[] pred_nominal=section4A.makePrediction(sets[0], sets[1], tab_classifiers[i]);
				double[] prediction=new double[pred_nominal.length];
				for (int j=0; j<pred_nominal.length; j++)
					prediction[j]=slices[(int) pred_nominal[j]];
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
	
}
