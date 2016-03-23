package weka;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.Random;

import org.apache.regexp.recompile;
import org.apache.spark.storage.TempBlockId;
import org.netlib.util.doubleW;
import org.netlib.util.intW;

import akka.dispatch.LoadMetrics;
import akka.japi.Util;
import tachyon.thrift.WorkerService.Processor.returnSpace;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;


public class Recommend 
{
	
	private Instances m_Data = null;
	private Classifier m_Classifier = null;
	public Recommend() throws FileNotFoundException, Exception
	{
		m_Classifier = (J48)SerializationHelper.read(new FileInputStream("model/J481217.model"));	
		String nameOfDataset = "female-love";
		FastVector attributes = new FastVector();
		attributes.addElement(new Attribute("birthday"));
		attributes.addElement(new Attribute("astrology"));
		attributes.addElement(new Attribute("height"));
		attributes.addElement(new Attribute("city"));
		attributes.addElement(new Attribute("newdegree"));
		attributes.addElement(new Attribute("marriage"));
		attributes.addElement(new Attribute("newincome"));

		FastVector classValues = new FastVector(2);
		classValues.addElement("0");
		classValues.addElement("1");
		attributes.addElement(new Attribute("Class", classValues));
		m_Data = new Instances(nameOfDataset, attributes, 10);
		m_Data.setClassIndex(m_Data.numAttributes()-1);
	}
	/**
	 * calcuate instace class
	 * @param dataInstince
	 * @throws Exception
	 */
	public double classifyInstance(double[] dataInstince) throws Exception 
	{
		Instances testSet = m_Data.stringFreeStructure();
		Instance instance = makeInstance(dataInstince,testSet);
//		System.out.println(m_Data.numAttributes());
//		System.out.println(instance);
		double predicted = m_Classifier.classifyInstance(instance);
//		System.out.println("predicted:"+predicted);
//		System.out.println("Message classified as : " +
//				m_Data.classAttribute().value((int)predicted));
		return predicted;
	}
	/**
	 * become data to Inatace format
	 * @param dataInstance
	 * @param data
	 * @return Instance
	 */
	private Instance makeInstance(double[] dataInstance,Instances data) 
	{
		Instance instance = new Instance(8);
		instance.setDataset(data);
		Attribute birthAtt = data.attribute("birthday");
		Attribute astroAtt = data.attribute("astrology");
		Attribute heigAtt = data.attribute("height");
		Attribute cityAtt = data.attribute("city");
		Attribute degreeAtt = data.attribute("newdegree");
		Attribute marrAtt = data.attribute("marriage");
		Attribute incomeAtt = data.attribute("newincome");
		
		instance.setValue(birthAtt, dataInstance[0]);
		instance.setValue(astroAtt, dataInstance[1]);
		instance.setValue(heigAtt, dataInstance[2]);
		instance.setValue(cityAtt, dataInstance[3]);
		instance.setValue(degreeAtt, dataInstance[4]);
		instance.setValue(marrAtt, dataInstance[5]);
		instance.setValue(incomeAtt, dataInstance[6]);
				
		return instance;
	}
	/**
	 * load model
	 * @param modelFile
	 * @return
	 * @throws FileNotFoundException
	 * @throws Exception
	 */
	public static J48 loadModel(String modelFile) throws FileNotFoundException, Exception
	{
		J48 j48 = (J48) SerializationHelper.read(new FileInputStream(modelFile));
		return j48;
	}
	/**
	 * predict Instance class
	 * @param recommend
	 * @param trainDataFile
	 * @throws Exception
	 */
	public void predictInstinceSet(Recommend recommend,String trainDataFile) throws Exception
	{
		Instances testInstances = null;
		DataSource source = new DataSource(trainDataFile);
		testInstances = source.getDataSet();
		
		for(int i=100;i<110;i++)
		{
			double[] testData = new double[7];
			testData[0] = testInstances.instance(i).value(0);
			testData[1] = testInstances.instance(i).value(1);
			testData[2] = testInstances.instance(i).value(2);
			testData[3] = testInstances.instance(i).value(3);
			testData[4] = testInstances.instance(i).value(4);
			testData[5] = testInstances.instance(i).value(5);
			testData[6] = testInstances.instance(i).value(6);
			recommend.classifyInstance(testData);
		}
	}
	/**
	 * load train data and become Instances format
	 * @param trainDataFile
	 * @return
	 * @throws Exception
	 */
	public static Instances loadTrainData(String trainDataFile) throws Exception
	{
		Instances trainData = null;
		DataSource source = new DataSource(trainDataFile);
		trainData = source.getDataSet();
		trainData.setClassIndex(trainData.numAttributes()-1);
		return trainData;
	}
	/**
	 * train model and save model to file
	 * @param classifier
	 * @param options
	 * @param trainDataFile
	 * @param saveToFile
	 * @throws Exception
	 */
	public void trainModel(Classifier classifier,String[] options,String trainDataFile,String saveToFile) throws Exception
	{
		Instances traindata = loadTrainData(trainDataFile);
		classifier.setOptions(options);
		classifier.buildClassifier(traindata);
		System.out.println(classifier.toString());
		
		if(new File(saveToFile).exists())
		{
			System.out.println("delete old model "+saveToFile);
			new File(saveToFile).delete(); 
		}
		
		SerializationHelper.write(saveToFile, classifier);
	}
	/**
	 * process inputData and make inputData become double[] 
	 * @param femaleAttributes
	 * @param maleAttributes
	 * @return processedData
	 * 
	 * note:inputData format
	 * femaleAttributes formating is [birthday,astrology,height,city,newdegree,marriage,newincome]
	 * maleAttributes formating is [birthday,astrology,height,city,newdegree,marriage,newincome]
	 * 
	 * birthday formating example: "1987" ,only need year
	 * city format require: serial number's length is equal or greater than 4 and start with "86"
	 */
	public double[] processInputData(String[] femaleAttributes,String[] maleAttributes)
	{
		double processedData[] = new double[femaleAttributes.length];
		for(int i=0;i<femaleAttributes.length;i++)
		{
			if(i==3) continue;
			processedData[i] = Double.parseDouble(femaleAttributes[i])-Double.parseDouble(maleAttributes[i]);
		}
		
		if(femaleAttributes[3].substring(0, 4).equals(maleAttributes[3].substring(0, 4)))
		{
			if(femaleAttributes[3].length()>=6&&maleAttributes[3].length()>=6)
			{
				if(femaleAttributes[3].substring(4, 6).equals(maleAttributes[3].substring(4, 6)))
					processedData[3] = 0;
				else
					processedData[3] = 5;	
			}
			else
				processedData[3] = 5;				
		}
		else 
		{
			processedData[3] = 10;
		}
		return processedData;	
	}
	
	
	public static void main(String[] args) throws FileNotFoundException, Exception 
	{
		Recommend recommend = new Recommend();
		RecommendEvalution recommendEvalution = new RecommendEvalution();
		J48 j48 = new J48();
		String classifierString = "weka.classifiers.trees.J48";
		String modelFile = "model/J481217.model";
		String trainDataFile = "data/recommend_female.arff";
		String saveToFile = "model/J481217.model";
		
		String[] options = weka.core.Utils.splitOptions("-C 0.3 -M 2");
//		recommend.predictInstinceSet(recommend,trainDataFile);
	
//		recommendEvalution.evalResult(recommend,modelFile,trainDataFile);
	
//		recommendEvalution.crossValidate(classifierString,trainDataFile,options,5);
//		recommend.trainModel(j48, options,trainDataFile,saveToFile);
		
		
		
	}
}
