package neur.learn;

import neur.NeuralException;
import neur.NeuralNet;
import neur.data.NeuralDataSet;

public abstract class LearningAlgorithm {
    
    protected NeuralNet neuralNet;
    
    public enum LearningMode {ONLINE,BATCH};
    
    protected enum LearningParadigm {SUPERVISED};
    
    protected LearningMode learningMode;
    
    protected LearningParadigm learningParadigm;
    
    protected int MaxEpochs=100; // Значение максимального количества эпох по умолчанию
    
    protected int epoch=0;
    
    protected double MinOverallError=0.001; // Значение минимальной общей ошибки по умлочанию
    
    protected double LearningRate=0.1;
    
    protected NeuralDataSet trainingDataSet;
    
    public boolean printTraining=false;
    
    public abstract void train() throws NeuralException;

    public abstract void forward() throws NeuralException;
    
    public abstract void forward(int i) throws NeuralException;

    public abstract void applyNewWeights();

    public abstract void print();
    
    public void setMaxEpochs(int _maxEpochs){
        this.MaxEpochs=_maxEpochs;
    }
    
    public int getEpoch(){
        return epoch;
    }
    
    public void setMinOverallError(Double _minOverallError){
        this.MinOverallError=_minOverallError;
    }
    
    public Double getMinOverallError(){
        return this.MinOverallError;
    }
    
    public void setLearningRate(Double _learningRate){
        this.LearningRate=_learningRate;
    }

    
}
