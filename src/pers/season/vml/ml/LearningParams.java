package pers.season.vml.ml;

public class LearningParams {
	public double initLearningRate = 0.1;
	public int learningRateCheckStep = 100;
	public double learningRateDescentRatio = 10;
	public double regularizationLambda = 1;
	public int batchSize = 1000;
	public int iteration = 1000; 
	
	public LearningParams () {
		
	}
	
}
