package repport;

import java.io.IOException;
import java.util.LinkedList;

import weka.classifiers.Classifier;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ConverterUtils.DataSource;

import common.IOR;
import common.Common;
import common.Progress;

public class section7A {
	public static void main(String[] args) throws Exception{
		createDataSet();
		computePrediction();
	}
	
	/* 
	 * This function use linear regression, also 
	 */
	public static void computePrediction() throws Exception{
		
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
		String[] labels_classifiers=new String[]{"LinearRegression","SVR","MLP"};
		
		String path=IOR.getPathRoot()+"generated_datasets/section7A/overall_prediction.arff";
		DataSource source_house = new DataSource(path);
		Instances data_house=source_house.getDataSet();
		
		Instances[] sets=section4A.createTrainingAndTestSet(data_house);
		for (int i=0; i<3; i++){
			double[] prediction=section4A.makePrediction(sets[0], sets[1], tab_classifiers[i]);
			IOR.saveDoubleArray(prediction, "generated_results/section7A", "prediction_"+labels_classifiers[i]+".txt");
		}
		
	}
	/*
	 * This function create a dataset with : 
	 * - consumption at time t
	 * - hour_of_the_day, day_of_the_week
	 * - consumption at time t-1, t-2, t-3, t-24, t-48, t-7*24
	 * - gradient at time t-1, t-2, t-3
	 * - Lagrangian at time t-1, t-2, t-3
	 * temperature
	 */
	public static void createDataSet() throws IOException{
		System.out.println("starting");
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
		
		// import the index of all interesting houses, to make the sum
		LinkedList<String> index_houses=IOR.loadFile("aggregated_data", "index.txt");
		// compute the overall consumption by summing data coming from every houses. 
		// notice that we also get the number of line of the file, and the timestamp.
		boolean firstFile=true;
		int[] timestamps=null;
		double[] overall_consum=null;
		int nb_lines=0;
		int count;
		System.out.println("determine overall consumption");
		Progress.initialize(1, index_houses.size());	
		for (String house : index_houses){
			Progress.showProgress(5);
			count=0;
			LinkedList<String> file=IOR.loadFile("aggregated_data", house+".txt");
			if (firstFile){
				nb_lines=file.size();
				timestamps=new int[nb_lines];
				overall_consum=new double[nb_lines];
			}
			for (String line : file){
				String[] values=line.split(",");
				overall_consum[count]+=Double.parseDouble(values[1]);
				if (firstFile)
					timestamps[count]=Integer.parseInt(values[0]);
				count++;
			}
			if (firstFile)
				firstFile=false;
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
		
		// create arff file (like for the over dataset, we start around 8 days after the beginning)
		System.out.println("create arff file ...");
		LinkedList<String> arff_file=new LinkedList<String>();
		arff_file.add(header);
		for (int t=8*24+1;t<nb_lines;t++){
			StringBuilder sb=new StringBuilder();
			sb.append(String.valueOf(round(overall_consum[t])));
			int[] ts=Common.transformTimeStamp(timestamps[t]);
			sb.append("," +ts[1]);
			sb.append("," +ts[2]);
			
			for (int i=1;i<4;i++){
				sb.append(","+String.valueOf(round(overall_consum[t-i])));
				sb.append(","+String.valueOf(round(overall_consum[t-i]-overall_consum[t-i-1])));
				sb.append(","+String.valueOf(round(overall_consum[t-i]+overall_consum[t-i-2]-2*overall_consum[t-i-1])));
			}
			
			sb.append(","+String.valueOf(round(overall_consum[t-24])));
			sb.append(","+String.valueOf(round(overall_consum[t-48])));
			sb.append(","+String.valueOf(round(overall_consum[t-7*24])));
			
			sb.append(","+String.valueOf(round(temps[t])));
			arff_file.add(sb.toString());
		}
		
		IOR.saveFile(arff_file, "generated_datasets/section7A", "overall_prediction.arff");
		System.out.println("done");
		
	}
	
	public static double round(double d){
		return Math.round(d*1000)/1000.0;
	}
}
