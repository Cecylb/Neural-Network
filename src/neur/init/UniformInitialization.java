package neur.init;

import neur.math.RandomNumberGenerator;

public class UniformInitialization extends WeightInitialization { //Ограничиваем генерацию весов пределами
    
    private double min;
    private double max;
    
    public UniformInitialization(double _min,double _max){
        this.min=_min;
        this.max=_max;
    }
    
    @Override
    public double Generate(){
        return RandomNumberGenerator.GenerateBetween(min, max);
    }
    
}
