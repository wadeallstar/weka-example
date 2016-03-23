package weka;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class RecommendEvalution {

	/**
	 * evaluation test set result
	 * @param recommend
	 * @param modelFile
	 * @param trainDataFile
	 * @throws Exception
	 */
	public void evalResult(Recommend recommend,String modelFile,String trainDataFile) throws Exception
	{
		Instances trainData = null;
		DataSource source = new DataSource(trainDataFile);
		trainData = source.getDataSet();
		trainData.setClassIndex(trainData.numAttributes()-1);
		Evaluation evaluation = new Evaluation(trainData);
		Classifier classifier = Recommend.loadModel(modelFile);
		evaluation.evaluateModel(classifier,trainData);
		System.out.println(evaluation.toSummaryString());
		System.out.println(evaluation.toMatrixString());
	}
	/**
	 * cross validate operation
	 * @param classifierString
	 * @param trainDataFile
	 * @param options
	 * @param numFolds
	 * @throws Exception
	 */
	public void crossValidate(String classifierString, String trainDataFile,String[] options,int numFolds) throws Exception
	{
		Instances trainData = null;
		DataSource source = new DataSource(trainDataFile);
		trainData = source.getDataSet();
		trainData.setClassIndex(trainData.numAttributes()-1);
		Evaluation evaluation = new Evaluation(trainData);
		Random random = new Random();
		evaluation.crossValidateModel(classifierString, trainData, 3, options, random);
		System.out.println(evaluation.toSummaryString());
	}
	public void evalOriginalData(Recommend recommend,String originalDataFile) throws IOException, Exception
	{
	 FileReader reader = new FileReader(originalDataFile);
	 BufferedReader br = new BufferedReader(reader);
	 String s1 = null;
	 double love = 0;
	 double refuse = 0;
	 while((s1 = br.readLine()) != null)
	  {
		  String[] originalData = s1.trim().split(",");
		  String[] femaleAttribute = new String[7];
		  String[] maleAttribute = new String[7];
		  
		  for(int i=0;i<7;i++)
		  {
			  femaleAttribute[i]=originalData[i];
		  }
		  for(int j=0;j<7;j++)
		  {
			  maleAttribute[j]=originalData[j+7];
		  }
		  double[] data = recommend.processInputData(femaleAttribute, maleAttribute);
		  
		  if(recommend.classifyInstance(data)==1)
		  {
			  love = love + 1;
		  }
		  else if(recommend.classifyInstance(data)==0)
		  {
			  refuse = refuse + 1;
		  }
	  }
	 br.close();
	 reader.close();
	 System.out.println("love:"+love);
	 System.out.println("refuse:"+refuse);
	}

	public static void main(String[] args) throws FileNotFoundException, Exception 
	{
		Recommend recommend = new Recommend();
		RecommendEvalution recommendEvalution = new RecommendEvalution();
		J48 j48 = new J48();
		String classifierString = "weka.classifiers.trees.J48";
		String modelFile = "model/J481217.model";
		String trainDataFile = "data/recommend_female.arff";
		String originalDataFile = "data/female_refuse.csv";
		
		String[] options = weka.core.Utils.splitOptions("-C 0.3 -M 2");
//		recommend.predictInstinceSet(recommend,trainDataFile);
		
		recommendEvalution.evalResult(recommend,modelFile,trainDataFile);
		
//		recommendEvalution.crossValidate(classifierString,trainDataFile,options,5);
		
		recommendEvalution.evalOriginalData(recommend, originalDataFile);
	}
}
