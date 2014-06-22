package repport;
import java.util.LinkedHashMap;

import common.IOR;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ConverterUtils.DataSource;


/*
 * This file realizes the optimization on the parameters  C and gamma or the SVR
 * for the data on aggregated values. 
 * we use epsilon based SVR regression, with fixed epsilon =0.01 
 */
public class SVMOptimizer1 {
	static String[] houses_used=new String[]{"1002","1014","1018","1022","1440","1695","4332", "1843","6568","1859","1807", "2103","2387","4879","2265","2945","3355","4076","4755","3816","1331","6445","5291","3660","1969"};

	public static void main(String[] args) throws Exception{
		double[] gammas=new double[]{0.001,0.01,0.1,1};
		double[] cs={1,10,100,1000,10000,100000,1000000};
		
		
		// build the indices of the validation set
		// we take 4/13 of the values as validation set
		int[] indices_val=new int[2550];
		int[] indices_train=new int[5850];
		int count_val=0,count_train=0;
		for (int i=0;i<8400;i++){
			if ((i%13==0 || i%13==3|| i%13==5|| i%13==8) && count_val<2550){
				indices_val[count_val]=i;
				count_val++;
			}
			else{
				indices_train[count_train]=i;
				count_train++;
			}
		}
		
		// build once for all the validation sets
		LinkedHashMap<String, Instances> training_sets=new LinkedHashMap<String, Instances>();
		LinkedHashMap<String, Instances> validation_sets=new LinkedHashMap<String, Instances>();
		for (String house : houses_used){
			DataSource source_house = new DataSource(IOR.getPathRoot()+"generated_datasets/section4B_oneHourAhead"+house+".arff");
			Instances data_house=source_house.getDataSet();
			// build the train and validation set
			Instances set_train=new Instances(data_house,0);
			for (int i : indices_train){
				set_train.add(data_house.instance(i));
			}
			System.out.println(set_train.numInstances());
			training_sets.put(house, set_train);
			Instances set_val=new Instances(data_house,0);
			for (int i : indices_val){
				set_val.add(data_house.instance(i));
			}
			System.out.println(set_val.numInstances());
			validation_sets.put(house, set_val);
		}
		
		for (double C : cs){
			for (double gamma : gammas){
			
				LibSVM svm=new LibSVM();
				svm.setSVMType(new SelectedTag(LibSVM.SVMTYPE_EPSILON_SVR, LibSVM.TAGS_SVMTYPE));
				svm.setGamma(gamma);
				svm.setCost(C);
				svm.setNormalize(true);
				svm.setEps(0.01);
				svm.setDebug(false);
				// the kernel is by default RBF
				
				double[] results=new double[houses_used.length+2];
				int count=0;
				System.out.println("Optimization for gamma="+String.valueOf(gamma)+" and C="+String.valueOf(C)+" started");
				// then we build a classifier for each house, and test it for each house
				for (String house : houses_used){
					System.out.println("house "+house +" started");
					Instances training_set=training_sets.get(house);
					training_set.setClassIndex(0);
					svm.buildClassifier(training_set);
					Evaluation eval = new Evaluation(training_sets.get(house));
					
					Instances validation_set=validation_sets.get(house);
					validation_set.setClassIndex(0);
					double RMSE=0;
					double power=0;
					for (int i=0;i<validation_set.numInstances();i++){
						double pred=eval.evaluateModelOnce(svm, validation_set.instance(i));
						double real=validation_set.instance(i).value(0);
						RMSE+=(pred-real)*(pred-real);
						power+=real*real;
					}
					RMSE/=validation_set.numInstances();
					power/=validation_set.numInstances();
					RMSE=Math.sqrt(RMSE);
					power=Math.sqrt(power);
					results[count]=RMSE/power;
					count++;
				}
				double mean=0;
				double std=0;
				for (int i=0;i<results.length-2;i++){
					mean+=results[i];
					std+=results[i]*results[i];
				}
				mean/=(results.length-2);
				std/=(results.length-2);
				std-=mean*mean;
				std=Math.sqrt(std);
				results[count]=mean;
				count++;
				results[count]=std;
				System.out.println("Optimization for gamma="+String.valueOf(gamma)+" and C="+String.valueOf(C)+" gives NRMSE = "+String.valueOf(mean));
				// save result in a file
				IOR.saveDoubleArray(results, "SVMOptimization", "gamma"+String.valueOf(gamma)+"andC"+String.valueOf(C)+",txt");
				
			}
		}
		
	}
	
}
