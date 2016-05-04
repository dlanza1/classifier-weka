package es.unex.mui.multimedia.classifiers;

import weka.core.Instances;

public class Utils {

	public static void removeAttribute(Instances instances, String name) {
		instances.deleteAttributeAt(instances.attribute(name).index());
	}
	
}
