package es.unex.mui.multimedia.classifiers;

import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Attribute;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToString;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.StringToNominal;

@SuppressWarnings("serial")
public class MultilayerPerceptronClassifier extends MultilayerPerceptron implements ClassifierInterface{

	public MultilayerPerceptronClassifier() {
		super();
		setLearningRate(0.3d);
		setMomentum(0.3);
		setNominalToBinaryFilter(true);
		setNormalizeNumericClass(true);
		//setTrainingTime(5000);
	}

	public Instances classifyInstances(Instances unlabeled) throws Exception {
		Instances labeled = new Instances(unlabeled);
		
		for (int i = 0; i < unlabeled.numInstances(); i++) {
			double clsLabel = classifyInstance(unlabeled.instance(i));
			labeled.instance(i).setClassValue(clsLabel);
		}
		
		return labeled;
	}

	public Instances preprocessTrainingData(Instances train) throws Exception {

		// Pasamos a Nominal los atributos Survived y Pclass
		NumericToNominal numericToNominalFilter = new NumericToNominal();
        String[] numericToNominalFilterOptions = {"-R", "2,3"};
        numericToNominalFilter.setOptions(numericToNominalFilterOptions);
        numericToNominalFilter.setInputFormat(train);
        train = Filter.useFilter(train, numericToNominalFilter);
        
        NominalToString nominalToStringFilter = new NominalToString();
        String[] nominalToStringFilterOptions = {"-C", "4"};
        nominalToStringFilter.setOptions(nominalToStringFilterOptions);
        nominalToStringFilter.setInputFormat(train);
        train = Filter.useFilter(train, nominalToStringFilter);
        
        // Quitamos Name
        Utils.removeAttribute(train, "Name");
        // Quitamos Cabin
        Utils.removeAttribute(train, "Cabin");
        // Quitamos PassengerId
        Utils.removeAttribute(train, "PassengerId");
        // Quitamos Ticket
        Utils.removeAttribute(train, "Ticket");
        
		//System.out.println(train);
        System.out.println(train.toSummaryString());
        
		return train;
	}
	
	public Instances preprocessTestingData(Instances unlabeled) throws Exception {
		
		StringToNominal stringToNominal = new StringToNominal();
		String[] options = {"-R", "8"};
		stringToNominal.setOptions(options);
        stringToNominal.setInputFormat(unlabeled);
        unlabeled = Filter.useFilter(unlabeled, stringToNominal);
        
        NumericToNominal numericToNominalFilter = new NumericToNominal();
        String[] numericToNominalFilterOptions = {"-R", "2"};
        numericToNominalFilter.setOptions(numericToNominalFilterOptions);
        numericToNominalFilter.setInputFormat(unlabeled);
        unlabeled = Filter.useFilter(unlabeled, numericToNominalFilter);
		
        List<String> values = new LinkedList<String>();
        values.add("0");
        values.add("1");
		unlabeled.insertAttributeAt(new Attribute("Survived", values ), 1);
		unlabeled.setClassIndex(1);
		
		// Quitamos Name
        Utils.removeAttribute(unlabeled, "Name");
        // Quitamos Cabin
        Utils.removeAttribute(unlabeled, "Cabin");
        // Quitamos PassengerId
        //Utils.removeAttribute(unlabeled, "PassengerId");
        // Quitamos Ticket
        Utils.removeAttribute(unlabeled, "Ticket");
		
		System.out.println(unlabeled.toSummaryString());
		
		return unlabeled;
	}

	public Evaluation evaluate(Instances train) throws Exception {
		Evaluation eval = new Evaluation(train);
		
		eval.crossValidateModel(this, train, 10, new Random(1));
		
		return eval;
	}

}
