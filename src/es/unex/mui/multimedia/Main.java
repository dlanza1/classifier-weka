package es.unex.mui.multimedia;

import java.io.PrintWriter;

import es.unex.mui.multimedia.classifiers.*;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Main {
	
	// Clasificador a utilizar
	enum Classifier {
		TREE {
			@Override
			ClassifierInterface get() {
				return new C45Classifier();
			}
		},
		BAYES {
			@Override
			ClassifierInterface get() {
				return new NaiveBayesClassifier();
			}
		},
		NETWORK {
			@Override
			ClassifierInterface get() {
				return new MultilayerPerceptronClassifier();
			}
		};
		
		abstract ClassifierInterface get();
	}
	
	private static Classifier DEFAULT_CLASSIFIER = Classifier.TREE;

	public static void main(String[] args) throws Exception {
		
		ClassifierInterface cl = null;
		if(args.length > 0 && args[0] != null)
			cl = Classifier.valueOf(args[0]).get();
		else
			cl = DEFAULT_CLASSIFIER.get();

		// Preparamos conjunto de datos de entrenamiento
		Instances train = new DataSource("data/train.csv").getDataSet(1);
		train = cl.preprocessTrainingData(train);
		
		// Entrenamos el clasificador
		cl.buildClassifier(train);
		
		// Evaluamos como de bueno es el clasificador
		// WARNING: overfitting
		Evaluation eval = cl.evaluate(train);
		System.out.println("=== Stratified cross-validation ===");
		System.out.println(eval.toSummaryString("\n=== Summary ===", true));
		System.out.println(eval.toClassDetailsString());
		System.out.println(eval.toMatrixString());
		
		// Preparamos conjunto de datos de test
		Instances unlabeled = new DataSource("data/test.csv").getDataSet();		
		unlabeled = cl.preprocessTestingData(unlabeled);
		
		// Clasificamos conjunto de test
		Instances labeled = cl.classifyInstances(unlabeled);
		
		// Escribimos resultados a un fichero
		PrintWriter writer = new PrintWriter("labeled.csv", "UTF-8");
		writer.println("PassengerId,Survived");
		for (Instance instance : labeled){
			writer.println(instance.toString(0) + "," + instance.toString(1));
		}
		writer.close();
	}
	
}
