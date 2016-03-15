import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import Jama.EigenvalueDecomposition;
import Jama.Matrix;


public class Utilities {
	
	//Input:	fullData is a list of training data. Each array in the list contains all the features; for n features, the data is in n dimensions.
	//Input:	testData is a single data point to classify agains the training data. The features should match up with fullData.
	//Output:	Returns training data sorted by distance from test data. 
	public static ArrayList<DistObj> performKNN(ArrayList<double[]> fullData, double[] testData) {
        int fullDataSize = fullData.size();

        ArrayList<DistObj> distObjects = new ArrayList<>();

        for (int i = 0; i < fullDataSize; i++) {
            double distance = calculateDistance(testData, fullData.get(i));
            DistObj dobj = new DistObj();
            dobj.index = i;
            dobj.distance = distance;
            distObjects.add(dobj);
        }

        sortDistObjs(distObjects);
        return distObjects;
	}
	
	//Calculates Euclidean distance in n-dimensional space (n is the size of the arrays).
    private static double calculateDistance(double[] array1, double[] array2) {
        double Sum = 0.0;
        for (int i = 0; i < array1.length; i++) {
            Sum = Sum + Math.pow((array1[i] - array2[i]), 2.0);
        }
        return Math.sqrt(Sum);
    }
    
    //Returns training data sorted by distance from test data. 
    //Smaller distances correspond to a more similar classification.
    private static void sortDistObjs(ArrayList<DistObj> distObjects) {
        Collections.sort(distObjects, new Comparator<DistObj>() {
            @Override
            public int compare(DistObj do1, DistObj do2) {
                return Double.compare(do1.distance, do2.distance);
            }
        });
    }
    
	public static double[][] getCovarianceMatrix(ArrayList<double[]> featureData, double[] dataAvg, int numFeatures) {

    	// create a copy of featureData
        ArrayList<double[]> featureDataAdjust = new ArrayList<>();
        for (double[] src : featureData) {
        	double[] dest = new double[numFeatures];
        	System.arraycopy( src, 0, dest, 0, src.length );
        	
        	featureDataAdjust.add(dest);
        }
        
        //creating data adjust
        for (double dataAdjustComps[] : featureDataAdjust) {
            for (int i = 0; i < numFeatures; i++) {
                dataAdjustComps[i] -= dataAvg[i];
            }
        }
        
        double[][] covarianceMatrix = new double[numFeatures][numFeatures];

        for (int i = 0; i < numFeatures; i++) {
            for (int j = 0; j < numFeatures; j++) {
                covarianceMatrix[i][j] = calculateCovariance(featureDataAdjust, i, j);
            }
        }
        
        return covarianceMatrix;
		
	}
	
	public static ArrayList<Double> getEigenvalueMatrix(double[][] covarianceMatrix, int numFeatures) {
		
        EigenvalueDecomposition evd = new EigenvalueDecomposition(new Matrix(covarianceMatrix));

        //eigenValues contains the eigenvalues of the covariance matrix
        double[] eigenValues = evd.getRealEigenvalues();
        
        //create a n ArrayList copy
        ArrayList<Double> eigenValuesList = new ArrayList<>(numFeatures);
        for (int i = 0; i < numFeatures; i++) {
            eigenValuesList.add(eigenValues[i]);
        }
        //sort the eigenvalues in descending order.
        //the largest eigenvalue corresponds with the most significant (highest variance) principal component.
        Collections.sort(eigenValuesList, new Comparator<Double>() {
            @Override
            public int compare(Double ev1, Double ev2) {
                double eigenVal1 = ev1.doubleValue();
                double eigenVal2 = ev2.doubleValue();
                return (eigenVal1 == eigenVal2) ? 0 : (eigenVal1 < eigenVal2 ? 1 : -1);
            }
        });
        
        return eigenValuesList;
	}
	
    public static ArrayList<double[]> calculatePCA(ArrayList<double[]> oldDataList, Matrix eigenVectors) {
    	int oldDataSize = oldDataList.size();
        int dataSize = oldDataList.get(0).length;
        
        double[][] oldData2dArray = new double[oldDataSize][dataSize];

        int count = 0;
        for (double dataLine[] : oldDataList) {
            System.arraycopy(dataLine, 0, oldData2dArray[count], 0, dataSize);
            count++;
        }
        
        Matrix oldDataMatrix = new Matrix(oldData2dArray);
        Matrix newDataMatrix = new Matrix(oldDataSize, dataSize);
        
        newDataMatrix = oldDataMatrix.times(eigenVectors);

        double[][] newData2dArray = new double[oldDataSize][dataSize];
        
        newData2dArray = newDataMatrix.getArrayCopy();
        ArrayList<double[]> newDataList = new ArrayList<>();

        for (int i = 0; i < oldDataSize; i++) {
            newDataList.add(newData2dArray[i]);
        }
        
        return newDataList;
    }
	
	public static Matrix getEigenvectorMatrix(double[][] covarianceMatrix, int numFeatures, int numFeaturesReduced) {
		
        List<EigenObject> eigenObjList = performEigenOperations(covarianceMatrix, numFeatures);

        double[][] eigenVector2dArray = new double[numFeatures][numFeaturesReduced];

        int eigenObjectCount = 0;

        for (EigenObject eigenObject : eigenObjList) {

            double[] eigenVector = eigenObject.getEigenVector();

            for (int i = 0; i < numFeatures && eigenObjectCount < numFeaturesReduced; i++) {
                eigenVector2dArray[i][eigenObjectCount] = eigenVector[i];
            }

            eigenObjectCount++;
        }
        
        Matrix eigenVectors = new Matrix(eigenVector2dArray);
        
        return eigenVectors;
	}
	
    private static List<EigenObject> performEigenOperations(double[][] covarianceMatrix, int dataSize) {
        Matrix evdMatrix = new Matrix(covarianceMatrix);
        EigenvalueDecomposition evd = new EigenvalueDecomposition(evdMatrix);

        double[] myEigenValues = new double[dataSize];

        double[][] myEigenVectorMatrixInput = new double[dataSize][dataSize];
        Matrix myEigenVectorMatrix = new Matrix(myEigenVectorMatrixInput);

        myEigenValues = evd.getRealEigenvalues();
        myEigenVectorMatrix = evd.getV();
        
        List<EigenObject> eigenObjList = new ArrayList<>(dataSize);
        for (int i = 0; i < dataSize; i++) {
            eigenObjList.add(new EigenObject(myEigenValues[i], myEigenVectorMatrix.getArray()[i]));
        }

        Collections.sort(eigenObjList, new Comparator<EigenObject>() {
            @Override
            public int compare(EigenObject eo1, EigenObject eo2) {
                double eigenVal1 = eo1.getEigenValue();
                double eigenVal2 = eo2.getEigenValue();
                return (eigenVal1 == eigenVal2) ? 0 : (eigenVal1 < eigenVal2 ? 1 : -1);
            }
        });

        return eigenObjList;
    }
	
    private static double calculateCovariance(ArrayList<double[]> fullDataAdjust, int i, int j) {

        double metricAdjustProdTotal = 0.0;  // the final numerator in the covariance formula i.e Summation[(Xi-Xmean)*(Yi-Ymean)]

        for (double dataAdjustComps[] : fullDataAdjust) {
            metricAdjustProdTotal += dataAdjustComps[i] * dataAdjustComps[j];
        }
        return metricAdjustProdTotal / (fullDataAdjust.size() - 1);
    }
}
