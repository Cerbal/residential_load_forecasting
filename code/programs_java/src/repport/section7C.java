package repport;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.LinkedList;

import weka.classifiers.Classifier;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ConverterUtils.DataSource;

import common.IOR;
import common.Common;;

public class section7C {
static int[] nb_clusters={1};
	
	static public void main(String[] args) throws Exception{
		createDataSet("trikmeans");
		predict("trikmeans");
	}
	
	static public void createDataSet(String clustering_type) throws IOException{
		// at first we extract all the data coming from the houses
		System.out.println("start create Dataset");
		
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
		for (int nb : nb_clusters){
			System.out.println("create arff file ...");
			
			// for each cluster
			LinkedList<String> clusters_description=IOR.loadFile("clusters/"+clustering_type, String.valueOf(nb)+".txt");
			int id_cluster=0;
			for (String cluster : clusters_description){
				double[] aggregated_consumption=new double[nb_lines];
				String[] houses=cluster.split(",");
				// we aggregate the consumption 
				for (String house : houses){
					double[] consum_h=consum_houses.get(house);
					for (int i=0;i<nb_lines;i++)
						aggregated_consumption[i]+=consum_h[i];						
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
				
				IOR.saveFile(arff_file, "clusters/"+clustering_type+"_arff_files/"+String.valueOf(nb)+"_clusters", "c"+String.valueOf(id_cluster)+".arff");
				id_cluster++;
				System.out.println("done");

			}
		}
	}
	
	static public void predict(String clustering_type) throws Exception{
		for (int nb : nb_clusters){
			System.out.println("\n start prediction for " +String.valueOf(nb)+" clusters\n");
			
			double[] overall_consum_linear=null;
			double[] overall_consum_mlp=null;
			double[] overall_consum_smoreg=null;
			double[] overall_consum_avg=null;
			for (int id_cluster=0;id_cluster<nb;id_cluster++){
				System.out.println("cluster "+String.valueOf(id_cluster)+"/" +String.valueOf(nb));
				
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
				String path=IOR.getPathRoot()+"clusters/"+clustering_type+"_arff_files/"+String.valueOf(nb)+"_clusters/c"+String.valueOf(id_cluster)+".arff";	
				DataSource source_house = new DataSource(path);
				Instances data_house=source_house.getDataSet();
				
				Instances[] sets=section4A.createTrainingAndTestSet(data_house);
				double[] pred_linear=section4A.makePrediction(sets[0], sets[1], tab_classifiers[0]);
				double[] pred_mlp=section4A.makePrediction(sets[0], sets[1], tab_classifiers[1]);
				double[] pred_smoreg=section4A.makePrediction(sets[0], sets[1], tab_classifiers[2]);
				if (overall_consum_linear==null){
					overall_consum_linear=new double[pred_linear.length];
					overall_consum_mlp=new double[pred_linear.length];
					overall_consum_smoreg=new double[pred_linear.length];
					overall_consum_avg=new double[pred_linear.length];
				}
				for (int i=0; i<pred_linear.length;i++){
					overall_consum_linear[i]+=pred_linear[i];
					overall_consum_mlp[i]+=pred_mlp[i];
					overall_consum_smoreg[i]+=pred_smoreg[i];
					overall_consum_avg[i]+=(pred_linear[i]+pred_mlp[i]+pred_smoreg[i])/3.0;
				}
			}
			IOR.saveDoubleArray(overall_consum_avg, "clusters/"+clustering_type+"_predictions/"+String.valueOf(nb)+"_clusters", "overall_prediction_avg.txt");
			IOR.saveDoubleArray(overall_consum_linear, "clusters/"+clustering_type+"_predictions/"+String.valueOf(nb)+"_clusters", "overall_prediction_linear.txt");
			IOR.saveDoubleArray(overall_consum_mlp, "clusters/"+clustering_type+"_predictions/"+String.valueOf(nb)+"_clusters", "overall_prediction_mlp.txt");
			IOR.saveDoubleArray(overall_consum_smoreg, "clusters/"+clustering_type+"_predictions/"+String.valueOf(nb)+"_clusters", "overall_prediction_smoreg.txt");
		}
	}
	
	public static double round(double d){
		return Math.round(d*1000)/1000.0;
	}

}
