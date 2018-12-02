package neur;

import neur.init.WeightInitialization;
import neur.math.IActivationFunction;
import java.util.ArrayList;

public abstract class NeuralLayer { // Класс абстрактный, инстациируется в InputLayer, HiddenLayer и OutputLayer

    protected int numberOfNeuronsInLayer;
    private ArrayList<Neuron> neuron;
    protected IActivationFunction activationFnc;
    protected NeuralLayer previousLayer;
    protected NeuralLayer nextLayer;
    protected ArrayList<Double> input;
    protected ArrayList<Double> output;
    protected int numberOfInputs;
    private NeuralNet neuralNet;

    public NeuralLayer(NeuralNet _neuralNet,int numberofneurons,IActivationFunction iaf){
        this.neuralNet=_neuralNet;
        this.numberOfNeuronsInLayer=numberofneurons;
        this.activationFnc=iaf;
        neuron = new ArrayList<>(numberofneurons);
        output = new ArrayList<>(numberofneurons);
    }

    public int getNumberOfNeuronsInLayer(){
        return numberOfNeuronsInLayer;
    }

    public ArrayList<Neuron> getListOfNeurons(){
        return neuron;
    }

    public NeuralLayer getPreviousLayer(){
        return previousLayer;
    }

    public NeuralLayer getNextLayer(){
        return nextLayer;
    }

    protected void setPreviousLayer(NeuralLayer layer){
        previousLayer=layer;
    }

    protected void setNextLayer(NeuralLayer layer){
        nextLayer=layer;
    }

    protected void init(WeightInitialization weightInitialization){
        if(numberOfNeuronsInLayer>=0){
            for(int i=0;i<numberOfNeuronsInLayer;i++){
                try{
                    neuron.get(i).setActivationFunction(activationFnc);
                    neuron.get(i).setNeuralLayer(this);
                    neuron.get(i).init(weightInitialization);
                }
                catch(IndexOutOfBoundsException iobe){
                    neuron.add(new Neuron(numberOfInputs,activationFnc));
                    neuron.get(i).setNeuralLayer(this);
                    neuron.get(i).init(weightInitialization);
                }
            }
        }
    }

    protected void setInputs(ArrayList<Double> inputs){
        this.numberOfInputs=inputs.size();
        this.input=inputs;
    }

    protected void calc(){
        if(input!=null && neuron!=null){
            for(int i=0;i<numberOfNeuronsInLayer;i++){
                neuron.get(i).setInputs(this.input);
                neuron.get(i).calc();
                try{
                    output.set(i,neuron.get(i).getOutput());
                }
                catch(IndexOutOfBoundsException iobe){
                    output.add(neuron.get(i).getOutput());
                }
            }
        }
    }

    protected ArrayList<Double> getOutputs(){
        return output;
    }

    public Neuron getNeuron(int i){
        return neuron.get(i);
    }

    protected void setNeuron(int i, Neuron _neuron){
        try{
            this.neuron.set(i, _neuron);
        }
        catch(IndexOutOfBoundsException iobe){
            this.neuron.add(_neuron);
        }
    }
    
    public Double getWeight(int i,int j){
        return this.neuron.get(j).getWeight(i);
    }
    
    public double[] getInputs(){
        double[] result = new double[numberOfInputs];
        for(int i=0;i<numberOfInputs;i++){
            result[i]=input.get(i);
        }
        return result;
    }
    
    public NeuralNet.NeuralNetMode getNeuralMode(){
        return this.neuralNet.getNeuralNetMode();
    }
    
}
