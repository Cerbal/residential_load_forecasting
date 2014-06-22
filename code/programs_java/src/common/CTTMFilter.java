package common;
import java.util.Arrays;
import java.util.LinkedList;

import weka.core.*;
 import weka.core.Capabilities.*;
import weka.filters.*;
 
 public class CTTMFilter
   extends SimpleBatchFilter {
	private static final long serialVersionUID = -4497074509347206780L;

	private int[] attributesToDiscretize;
	private int nb_bits;
	double[][] bounds;
	public double[][] val_interval;
	
	public CTTMFilter(int nb_bits, String attributesToDiscretize) throws Exception {
		attributesToDiscretize=attributesToDiscretize.trim();
		this.nb_bits=nb_bits;
		String[] portions=attributesToDiscretize.split(",");
		LinkedList<Integer> attributes=new LinkedList<Integer>();
		try{
			for (String s  : portions){
				if (s.contains("-")){
					String[] bounds=s.split("-");
					int start=Integer.parseInt(bounds[0]);
					int end=Integer.parseInt(bounds[1]);
					for (int i=start; i<=end; i++){
						attributes.add(i);
					}
				}
				else{
					attributes.add(Integer.parseInt(s));
				}
			}
			this.attributesToDiscretize=new int[attributes.size()];
			int count=0;
			for (Integer i : attributes){
				this.attributesToDiscretize[count]=i;
				count++;
			}
		}
		catch(Exception e){
			throw new Exception("Unable to cast the columns definition of the CTTM FIlter");
		}
	}
	
	public CTTMFilter(int nb_bits, int[] attributesToDiscretize){
		super();
		this.nb_bits=nb_bits;
		this.attributesToDiscretize=attributesToDiscretize;
	}
	
	public String globalInfo() {
     return   "A filter that discretizes values of each attribute according to their contribution to the mean";
   }
 
   public Capabilities getCapabilities() {
     Capabilities result = super.getCapabilities();
     result.enableAllAttributes();
     result.enableAllClasses();
     result.enable(Capability.NO_CLASS);  //// filter doesn't need class to be set//
     return result;
   }
 
   protected Instances determineOutputFormat(Instances inputFormat) {
     Instances result = new Instances(inputFormat, 0);
     FastVector poss_attributes=new FastVector();
     for (int i=0;i<nb_bits;i++){
    	 poss_attributes.addElement(String.valueOf(i));
     }
     for (int i=0;i<attributesToDiscretize.length;i++){
    	 String att_name=new String(result.attribute(attributesToDiscretize[i]).name());
    	 result.deleteAttributeAt(attributesToDiscretize[i]);
    	 result.insertAttributeAt(new Attribute(att_name, poss_attributes), attributesToDiscretize[i]);
     }
     return result;
   }
 
   public void establishBounds(Instances inst){
	     // for every attribute, we determine first the bounds, that will make possible 
	     // discretion. We determine also the value associated with each interval
	     bounds=new double[attributesToDiscretize.length][nb_bits-1];
	     val_interval=new double[attributesToDiscretize.length][nb_bits];
	     for (int i=0;i<attributesToDiscretize.length;i++){
	    	 double[] values=inst.attributeToDoubleArray(attributesToDiscretize[i]);
	    	 // we calculate the sum of the data
	    	 double sum=0;
	    	 Arrays.sort(values);
	    	 for (int j=0;j<values.length;j++){
	    		 sum+=values[j];
	    	 }
	    	 double stop_bounds=sum/nb_bits;
	    	 int num_bound=0;
	    	 int element_in_interval=0;
	    	 // then determine bounds
	    	 double temp_sum=0;
	    	 for (int j=0;j<values.length;j++){
	    		 element_in_interval++;
	    		 temp_sum+=values[j];
	    		 if (temp_sum>stop_bounds && num_bound<nb_bits-1){
	    			 bounds[i][num_bound]=values[j];
	    			 val_interval[i][num_bound]=sum/nb_bits/element_in_interval;
	    			 element_in_interval=0;
	    			 stop_bounds+=sum/nb_bits;
	    			 num_bound++;
	    		 }
	    	 }
	    	 val_interval[i][nb_bits-1]=sum/nb_bits/element_in_interval;
	     }
   }
   
   public Instances process(Instances inst) {
     Instances result = new Instances(determineOutputFormat(inst), 0);
 
     // then fill the new Instances with our discretized dataset
     for (int i = 0; i < inst.numInstances(); i++) {
       double[] values = new double[result.numAttributes()];
       // by default we fill the attributes like they were
       for (int n = 0; n < inst.numAttributes(); n++)
         values[n] = inst.instance(i).value(n);
       for (int n=0;n<attributesToDiscretize.length;n++){
    	   values[attributesToDiscretize[n]]=(double) discretize(values[attributesToDiscretize[n]], n);
       }
       result.add(new Instance(1, values));
     }
     return result;
   }
   
 
   public int discretize(double value, int att){
	   for (int i=0;i<bounds[att].length;i++){
		   if (bounds[att][i]>value)
			   return i;
	   }
	   return bounds[att].length;
   }

 }
