package neur.math;

public class Sigmoid implements IActivationFunction {

    private double a=1.0;

    public Sigmoid(double value){
        this.setA(value);
    }

    public void setA(double value){
        this.a=value;
    }

    @Override
    public double calc(double x){
        return 1.0/(1.0+Math.exp(-a*x));
    }
    
    @Override 
    public double derivative(double x){
        return calc(x)*(1-calc(x));
    }
}
