package neur.math;

import java.util.Random;

public class RandomNumberGenerator {

    public static long seed=0;

    public static Random r;
    
    public static double GenerateBetween(double min,double max){
        if(r==null)
            r=new Random(seed);
        if(max<min)
           return min;
        return min+(r.nextDouble()*(max-min));
    }
}
