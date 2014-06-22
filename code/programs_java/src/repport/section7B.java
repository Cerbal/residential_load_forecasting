package repport;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import common.IOR;
import common.Common;


import weka.classifiers.Classifier;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ConverterUtils.DataSource;

public class section7B {
	/*

	 */
static int[] nb_houses_in_clusters={782};
static int N_redondance_max=2;

public static void main(String[] args) throws Exception{
	createDataSets();
	predict();
}

/*
 * This function takes randoms houses, aggregate them and create dataset for the aggregation
 */
static public void createDataSets() throws IOException{
	// at first we extract all the data coming from the houses
	System.out.println("start create Datasets");
	
	// creation of header
	String header="@RELATION OverallConsumption\n\n";
	header+="@ATTRIBUTE consum_time_t NUMERIC\n";
	header+="@ATTRIBUTE day_week NUMERIC\n";
	header+="@ATTRIBUTE hour_day NUMERIC\n";
	for (int i=1;i<4;i++){
		header+="@ATTRIBUTE consum_time_t_"+i+" NUMERIC\n";
		header+="@ATTRIBUTE gradient_time_t_"+i+" NUMERIC\n";
		header+="@ATTRIBUTE lagrangian_time_t_"+i+" NUMERIC\n";
	}
	header+="@ATTRIBUTE consum_time_t_24 NUMERIC\n";
	header+="@ATTRIBUTE consum_time_t_48 NUMERIC\n";
	header+="@ATTRIBUTE consum_day_7 NUMERIC\n";
	header+="@ATTRIBUTE temperature NUMERIC\n";
	header+="\n\n @DATA";
	
	// as usual
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
	count=0;
	for(String line : ls_temp){
		temps[count]=Double.parseDouble(line);
		count++;
	}
	
	// acquire datasets
	System.out.println("acquiring the whole dataset");
	System.out.println("consumption");
	LinkedHashMap<String,double[]> consum_houses=IOR.crawlHouses(index_houses, "aggregated_data", nb_lines, true);
	
	// for each number of clusters
	for (int nb : nb_houses_in_clusters){
		System.out.println("for "+String.valueOf(nb));
		for (int red=0;red<N_redondance_max;red++){
		System.out.println("red"+String.valueOf(red));
		
		ArrayList<Integer> indices=new ArrayList<Integer>();
		for (int i=0;i<782;i++)
			indices.add(i);
		Collections.shuffle(indices);
		// take nb houses randomly, and aggregate
		double[] aggregated_consumption=new double[nb_lines];
		
		for (int i=0;i<nb;i++){
			double[] consum=consum_houses.get(index_houses.get(indices.get(i)));
			for (int j=0;j<nb_lines;j++){
				aggregated_consumption[j]+=consum[j];
			}
		}

		
		// and then create the arff file
		LinkedList<String> arff_file=new LinkedList<String>();
		arff_file.add(header);
		for (int t=8*24+1;t<nb_lines;t++){
			StringBuilder sb=new StringBuilder();
			sb.append(String.valueOf(round(aggregated_consumption[t])));
			int[] ts=Common.transformTimeStamp(timestamps[t]);
			sb.append("," +ts[1]);
			sb.append("," +ts[2]);
			
			for (int i=1;i<4;i++){
				sb.append(","+String.valueOf(round(aggregated_consumption[t-i])));
				sb.append(","+String.valueOf(round(aggregated_consumption[t-i]-aggregated_consumption[t-i-1])));
				sb.append(","+String.valueOf(round(aggregated_consumption[t-i]+aggregated_consumption[t-i-2]-2*aggregated_consumption[t-i-1])));
			}
			
			sb.append(","+String.valueOf(round(aggregated_consumption[t-24])));
			sb.append(","+String.valueOf(round(aggregated_consumption[t-48])));
			sb.append(","+String.valueOf(round(aggregated_consumption[t-7*24])));
			sb.append(","+temps[t]);
			arff_file.add(sb.toString());
		}
		
		IOR.saveFile(arff_file, "generated_datasets/section7B/"+String.valueOf(nb)+"houses/", "red"+String.valueOf(red)+".arff");

		}
	}
	System.out.println("done");
}

static public void predict() throws Exception{
	// for each number of clusters
	for (int nb : nb_houses_in_clusters){
		System.out.println("starting prediction one hour ahead for "+String.valueOf(nb)+" houses");
		for (int red=0;red<N_redondance_max;red++){
	
			System.out.println("red"+String.valueOf(red));
			System.out.println("\n start prediction for " +String.valueOf(nb)+" clusters\n");
			
			// a linear regression
			LinearRegression linreg=new LinearRegression();
			
			// an MLP
			MultilayerPerceptron mlp=new MultilayerPerceptron();
			mlp.setValidationSetSize(30);
			mlp.setValidationThreshold(20);
			
			// and a SVR, with the optimum gamma and C found in the file sVM classifier
			LibSVM svm= new LibSVM();
			svm.setSVMType(new SelectedTag(LibSVM.SVMTYPE_EPSILON_SVR, LibSVM.TAGS_SVMTYPE));
			svm.setGamma(1);
			svm.setCost(1000);
			svm.setNormalize(true);
			svm.setEps(0.01);
			svm.setDebug(false);
			
			Classifier tab_classifiers[]=new Classifier[]{linreg, svm, mlp};
			String path=IOR.getPathRoot()+"generated_datasets/section7B/"+String.valueOf(nb)+"houses/red"+String.valueOf(red)+".arff";
			DataSource source_house = new DataSource(path);
			Instances data_house=source_house.getDataSet();
			
			Instances[] sets=section4A.createTrainingAndTestSet(data_house);
			double[] pred_linear=section4A.makePrediction(sets[0], sets[1], tab_classifiers[0]);
			double[] pred_mlp=section4A.makePrediction(sets[0], sets[1], tab_classifiers[1]);
			double[] pred_svr=section4A.makePrediction(sets[0], sets[1], tab_classifiers[2]);
			double[] reals=sets[1].attributeToDoubleArray(0);


			IOR.saveDoubleArray(pred_linear, "generated_results/section7B/"+String.valueOf(nb)+"houses/", "red"+String.valueOf(red)+"_prediction_linear.txt");
			IOR.saveDoubleArray(pred_mlp, "generated_results/section7B/"+String.valueOf(nb)+"houses/", "red"+String.valueOf(red)+"_prediction_mlp.txt");
			IOR.saveDoubleArray(pred_svr, "generated_results/section7B/"+String.valueOf(nb)+"houses/", "red"+String.valueOf(red)+"_prediction_svr.txt");
			IOR.saveDoubleArray(reals, "generated_results/section7B/"+String.valueOf(nb)+"houses/", "red"+String.valueOf(red)+"_reals.txt");
		}
	}
}

public static double round(double d){
	return Math.round(d*1000)/1000.0;
}

/*
 * This is just to correct my error and rename some of the file to .txt
 */
public static void renameIntoTxt() throws IOException{
	for (int nb : nb_houses_in_clusters){
		System.out.println("starting prediction one hour ahead for "+String.valueOf(nb)+" houses");
		for (int red=0;red<N_redondance_max;red++){
			   File old_name = new File(IOR.getPathRoot()+"linearVsSVR/"+String.valueOf(nb)+"houses/red"+String.valueOf(red)+"_reals.arff");
			   File new_name = new File(IOR.getPathRoot()+"linearVsSVR/"+String.valueOf(nb)+"houses/red"+String.valueOf(red)+"_reals.txt");
			   old_name.renameTo(new_name);
		}
	}
}

}
