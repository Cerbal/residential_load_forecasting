package repport;

import common.IOR;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ConverterUtils.DataSource;


/*
 * This file confirm that parameter epsilon does not influence results
 * Hypothesis of independance with gamma and C has been made, and so we verified only 
 * with gamma =1 and C =1000, optimum parameters found for epsilon =0.01.
 * 
 * 
 */
public class SVMOptimizer3 {
	public static void main(String[] args) throws Exception{
		double[] epss={0.000003,0.00001,0.00003,0.0001,0.0003,0.001};
		
		
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
		DataSource source = new DataSource(IOR.getPathRoot()+"generated_datasets/section7A/overall_prediction.arff");
		Instances data=source.getDataSet();
		// build the train and validation set
		Instances set_train=new Instances(data,0);
		for (int i : indices_train){
			set_train.add(data.instance(i));
		}
		System.out.println(set_train.numInstances());
		Instances set_val=new Instances(data,0);
		for (int i : indices_val){
			set_val.add(data.instance(i));
		}
		System.out.println(set_val.numInstances());

		int nbtestes=epss.length;
		double[] results=new double[nbtestes];
		int count=0;
		for (double epsilon:epss){
				LibSVM svm=new LibSVM();
				svm.setSVMType(new SelectedTag(LibSVM.SVMTYPE_EPSILON_SVR, LibSVM.TAGS_SVMTYPE));
				svm.setGamma(1);
				svm.setCost(1000);
				svm.setNormalize(true);
				svm.setEps(epsilon);
				svm.setDebug(false);
				// the kernel is by default RBF


				System.out.println("Optimization for epsilon="+String.valueOf(epsilon));
				// then we build a classifier for each house, and test it for each house

				Instances training_set=set_train;
				training_set.setClassIndex(0);
				svm.buildClassifier(training_set);
				Evaluation eval = new Evaluation(set_train);
				
				Instances validation_set=set_val;
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
				double NRMSE=RMSE/power;
				
				results[count]=NRMSE;
				count++;

				System.out.println("Optimization for epsilon="+String.valueOf(epsilon)+" gives NRMSE = "+String.valueOf(NRMSE));
		}
		IOR.saveDoubleArray(results, "SVMOptimization", "epsilons.txt");
	}
	
}
