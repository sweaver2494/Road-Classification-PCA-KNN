/*
 * Perform KNN using k = K_VALUE on original data to find error benchmark
 * 
 * Feature Test:
 * N=total number of features
 * Perform KNN to find error again using 1 to N-1 features, in order of variance. Feature 1 has the most variance, feature N has the least.
 * Start with all features, the N-1 features, then N-2 features, until only feature 1 is left.
 * 
 * Component Test:
 * N=total number of features
 * Perform PCA using 1 to N principal components, in order of variance, each time performing KNN again to get error.
 *
 * @author Scott Weaver
 */
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;

import Jama.Matrix;

public class PcaKnn {

	private static String FEATURE_LIST_PATH = "Data/feature_list.txt";
	private static String TRAINING_FILE_PATH = "Data/TrainingData/training_data.csv";
    //Must include. This is the file you want to test against feature files
    private static String TEST_FILE_PATH = "Data/TestData/test_data.csv";
    private static String RESULTS_FILE_PATH = "Data/results.csv";
    //Knn will be performed using k = K_VALUE
    private static int K_VALUE = 3;
    
	public static void main(String[] args) {
		
		System.out.println("Training File Path: " + TRAINING_FILE_PATH);
		System.out.println("Test File Path: " + TEST_FILE_PATH);
		
		ArrayList<double[]> trainingData = new ArrayList<>();
		ArrayList<double[]> testData = new ArrayList<>();
		ArrayList<String> trainingClassification = new ArrayList<>();
		ArrayList<String> testClassification = new ArrayList<>();
		//featureList is a list of features in ascending order of variance
		ArrayList<String> featuresOrderedByVariance = new ArrayList<>();
		ArrayList<String> features = new ArrayList<>();
		ArrayList<String> resultsFeature = new ArrayList<>();
		ArrayList<String> resultsComponent = new ArrayList<>();
		
		if ((new File(TEST_FILE_PATH)).isFile() && (new File(TRAINING_FILE_PATH)).isFile() && (new File(FEATURE_LIST_PATH)).isFile()) {
			String trainingColumnHeaders = readFeatureFile(TRAINING_FILE_PATH, trainingData, trainingClassification);
			String testColumnHeaders = readFeatureFile(TEST_FILE_PATH, testData, testClassification);
			
			if (trainingColumnHeaders.equals(testColumnHeaders)) {
				readFeatureListFile(FEATURE_LIST_PATH, featuresOrderedByVariance);
				Collections.reverse(featuresOrderedByVariance);
				features = getColumnHeaders(testColumnHeaders);
				
				performPrincipalFeatureTest(trainingData, trainingClassification, testData, testClassification, features, featuresOrderedByVariance, resultsFeature);
				
				performPrincipalComponentTest(trainingData, trainingClassification, testData, testClassification, features, resultsComponent);
				
				writeResultsFile(RESULTS_FILE_PATH, resultsFeature, resultsComponent);
			} else {
				System.err.println("Features inconsistent between Test File and Training File (mismatching headers).");
			}
		} else {
			System.err.println("Test File or Training File does not exist.");
		}
	}
	
	private static void performPrincipalFeatureTest(ArrayList<double[]> trainingData, ArrayList<String> trainingClassification, ArrayList<double[]> testData, ArrayList<String> testClassification, ArrayList<String> features, ArrayList<String> featuresOrderedByVariance, ArrayList<String> results) {
		int numTestData = testData.size();
		int numTrainingData = trainingData.size();
		int numFeatures = featuresOrderedByVariance.size();
		int numRemoved = 0;
		
		//MAKE A COPY OF TRAINING AND TEST DATA FIRST
        ArrayList<double[]> trainingDataCopy = arrayDeepCopy(trainingData, numTestData);
        ArrayList<double[]> testDataCopy = arrayDeepCopy(testData, numTrainingData);
        ArrayList<String> featuresCopy = new ArrayList<>(numFeatures);
        for (String str : features) {
        	featuresCopy.add(str);
        }
        
		for (String featureToRemove : featuresOrderedByVariance) {
			System.out.println("-----------------------------------");
			System.out.println("Removed Feature: " + featureToRemove);
			System.out.println();
			System.out.println("KNN Classification Accuracy");
	        
			//For each test data point, perform KNN using k=K_VALUE
			int count = 0;
			for (int i = 0; i < numTestData; i++) {
				double[] test = testDataCopy.get(i);
				String actualClassification = testClassification.get(i);
				
				ArrayList<DistObj> distanceObjects = Utilities.performKNN(trainingDataCopy, test);
				
				String predictedClassification = getPredictedClassification(distanceObjects, trainingClassification, K_VALUE);
				
				if (actualClassification.equals(predictedClassification)) {
					count++;
				}	
			}
			
			System.out.println();
			System.out.println("KNN Feature Accuracy");
			//print out the percentage of correctly classified tests using that k for KNN.
			
			double acc = ((double) count) / numTestData;
			System.out.println(numRemoved + " / " + numFeatures + " removed:\t" + count + "/" + numTestData + " = " + acc);
			System.out.println();
			results.add(String.valueOf(acc));
			
			//REMOVE FEATURE AT INDEX featureToRemove
			int indexToRemove = 0;
			for (String feature : featuresCopy) {
				if (feature.equals(featureToRemove)) {
					break;
				}
				indexToRemove++;
			}
			featuresCopy.remove(indexToRemove);			
			arrayRemoveFeature(trainingDataCopy, numTrainingData, indexToRemove);
			arrayRemoveFeature(testDataCopy, numTestData, indexToRemove);
	        numRemoved++;
		}
	}
	
	private static void performPrincipalComponentTest(ArrayList<double[]> trainingData, ArrayList<String> trainingClassification, ArrayList<double[]> testData, ArrayList<String> testClassification, ArrayList<String> features, ArrayList<String> results) {
		int numFeatures = features.size();
		int numTestData = testData.size();
		double dataAverage[] = new double[numFeatures];
		
		getAverage(trainingData, dataAverage);
    	
		double[][] covarianceMatrix = Utilities.getCovarianceMatrix(trainingData, dataAverage, numFeatures);
	            
		int numRemoved = 0;
        
		for (int numComponents = numFeatures; numComponents > 0; numComponents--) {
			System.out.println("-----------------------------------");
			System.out.println("Removed Component: " + numComponents);
			System.out.println();
			System.out.println("KNN Classification Accuracy");
			
			Matrix eigenVectorMatrix = Utilities.getEigenvectorMatrix(covarianceMatrix, numFeatures, numComponents);
	    	//use eigenvectors to calculate the reduced data set
	        ArrayList<double[]> newTrainingData = Utilities.calculatePCA(trainingData, eigenVectorMatrix);
	        //use the same eigenvectors to reduce the test data to fit in the same dimensionality as the training data
	        ArrayList<double[]> newTestData = Utilities.calculatePCA(testData, eigenVectorMatrix);
	        
			//For each test data point, perform KNN using k=K_VALUE
			int count = 0;
			for (int i = 0; i < numTestData; i++) {
				double[] test = newTestData.get(i);
				String actualClassification = testClassification.get(i);
				
				ArrayList<DistObj> distanceObjects = Utilities.performKNN(newTrainingData, test);
				
				String predictedClassification = getPredictedClassification(distanceObjects, trainingClassification, K_VALUE);
				
				if (actualClassification.equals(predictedClassification)) {
					count++;
				}	
			}
			
			System.out.println();
			System.out.println("KNN Component Accuracy");
			//print out the percentage of correctly classified tests using that k for KNN.
			
			double acc = ((double) count) / numTestData;
			System.out.println(numRemoved + " / " + numFeatures + " removed:\t" + count + "/" + numTestData + " = " + acc);
			System.out.println();
			results.add(String.valueOf(acc));
			
			numRemoved++;
			
		}
	}
	
	//dataAverage will contain the average value for each feature (column)
	private static void getAverage(ArrayList<double[]> featureData, double[] dataAverage) {
		int dataSize = featureData.size();
		int numFeatures = dataAverage.length;
		
		for (double[] data : featureData) {
			for (int i = 0; i < numFeatures; i++) {
				dataAverage[i] += data[i];
			}
		}
		
		for (int i = 0; i < numFeatures; i++) {
			dataAverage[i] /= dataSize;
		}
	}
	
	private static ArrayList<double[]> arrayDeepCopy(ArrayList<double[]> original, int size) {
		ArrayList<double[]> copy = new ArrayList<>(size);
		
        for (double[] src : original) {
        	double[] dest = new double[src.length];
        	System.arraycopy( src, 0, dest, 0, src.length );
        	
        	copy.add(dest);
        }
        
        return copy;
	}
	
	private static void arrayRemoveFeature(ArrayList<double[]> list, int size, int indexToRemove) {
        for (int i = 0; i < size; i++) {
        	double[] src = list.remove(i);
            
        	double[] dest = new double[src.length-1];
        	System.arraycopy( src, 0, dest, 0, indexToRemove );
        	System.arraycopy( src, indexToRemove+1, dest, indexToRemove, src.length - 1 - indexToRemove );
        	
        	list.add(i, dest);
        }
	}
	
	//Return the results of KNN (predicted classification) using k. Print out the percentage of training data that the test data matched up with
	private static String getPredictedClassification(ArrayList<DistObj> distanceObjects, ArrayList<String> trainingClassification, int k) {
		
		HashMap<String,Integer> numOccurances = new HashMap<>();
		for (int i = 0; i < k; i++) {
			int index = distanceObjects.get(i).index;
			String classification = trainingClassification.get(index);
			
			Integer count = numOccurances.get(classification);
			numOccurances.put(classification, count==null?1:count+1);
		}
		
		String classification = "";
		int max = 0;
		
		for (String key : numOccurances.keySet()) {
			int val = numOccurances.get(key);
			
			if (val > max) {
    			max = val;
    			classification = key;
    		}
		}
		
		System.out.println(classification + ":\t\t" + max + "/" + k);
		
		return classification;
		
	}
	
    private static ArrayList<String> getColumnHeaders(String line) {
    	ArrayList<String> columnHeaders = new ArrayList<>();
    	
    	String dataCompsStr[] = line.substring(line.indexOf(",") + 1).split(",");
    	
    	for (String feature : dataCompsStr) {
    		columnHeaders.add(feature);
    	}
    	
    	return columnHeaders;
    }
	
	private static String readFeatureFile(String featureFilePath, ArrayList<double[]> featureData, ArrayList<String> featureClassification) {
		String columnHeaders = "";
		
		try {

	        BufferedReader bufferedReader = new BufferedReader(new FileReader(featureFilePath));
	        
	        columnHeaders = bufferedReader.readLine();

	        String line = bufferedReader.readLine();
	        int dataSize = line.length() - line.replace(",", "").length();
	
	        while (line != null) {
	        	String classification = line.substring(0, line.indexOf(","));
	            String dataCompsStr[] = line.substring(line.indexOf(",") + 1).split(",");
	
	            double dataComps[] = new double[dataSize];
	
	            for (int i = 0; i < dataSize; i++) {
	                dataComps[i] = Double.parseDouble(dataCompsStr[i]);
	            }
	
	            featureData.add(dataComps);
	            featureClassification.add(classification);
	            line = bufferedReader.readLine();
	        }
	        bufferedReader.close();
	        
        } catch(IOException e) {
        	System.err.println("Cannot read feature file.");
        }
		
		return columnHeaders;
	}
	
	private static void readFeatureListFile(String featureListPath, ArrayList<String> featureList) {
		try {

	        BufferedReader bufferedReader = new BufferedReader(new FileReader(featureListPath));

	        String line = bufferedReader.readLine();
	
	        while (line != null) {
	        	featureList.add(line);
	        	
	            line = bufferedReader.readLine();
	        }
	        bufferedReader.close();
	        
        } catch(IOException e) {
        	System.err.println("Cannot read feature list file.");
        }
	}
	
	private static void writeResultsFile(String filePath, ArrayList<String> resultsFeature, ArrayList<String> resultsComponent) {
		
		try {
			BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(filePath, true));
			
			bufferedWriter.write("Num Removed,Features,Components");
			bufferedWriter.newLine();
			for (int i = 0; i < resultsFeature.size(); i++) {
				bufferedWriter.write(i + "," + resultsFeature.get(i) + "," + resultsComponent.get(i));
				bufferedWriter.newLine();
			}
			bufferedWriter.close();
		} catch (IOException e) {
			System.err.println("Cannot write results file. " + e.getMessage());
		}
	}

}
