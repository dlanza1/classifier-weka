package es.unex.mui.multimedia;

import weka.classifiers.Evaluation;
import weka.core.Instances;

public interface ClassifierInterface {

	public Instances classifyInstances(Instances unlabeled) throws Exception;
	
	public Instances preprocessTrainingData(Instances train) throws Exception;
	
	public Instances preprocessTestingData(Instances unlabeled) throws Exception;
	
	public Evaluation evaluate(Instances train) throws Exception;
	
	public void buildClassifier(Instances instances) throws Exception;
	
}
