/*
 *
 * @author Scott Weaver
 */
public class EigenObject {
    private double eigenValue;

    public double getEigenValue() {
        return eigenValue;
    }

    public void setEigenValue(double eigenValue) {
        this.eigenValue = eigenValue;
    }

    public double[] getEigenVector() {
        return eigenVector;
    }

    public void setEigenVector(double[] eigenVector) {
        this.eigenVector = eigenVector;
    }
    private double[] eigenVector;
    
    public EigenObject(double eigenValue, double[] eigenVector){
    
        this.eigenValue = eigenValue;
        this.eigenVector = eigenVector;
        
    }
}