package neur;

import neur.init.UniformInitialization;
import neur.init.WeightInitialization;
import neur.math.IActivationFunction;
import java.util.ArrayList;

public class Neuron {

    protected ArrayList<Double> weight; // Массив весов
    private ArrayList<Double> input; // Массив входных значений
    private Double output; // Выходное значение
    private Double outputBeforeActivation; //Значение, с которым будет работать ActivationFunction
    private int numberOfInputs = 0; // Число входных значений
    protected Double bias = 1.0; // Значение биаса
    private IActivationFunction activationFunction;
    private NeuralLayer neuralLayer;
    private Double firstDerivative;

    public Neuron(int numberofinputs){
        numberOfInputs=numberofinputs;
        weight=new ArrayList<>(numberofinputs+1);
        input=new ArrayList<>(numberofinputs);
    }

    public Neuron(int numberofinputs,IActivationFunction iaf){
        numberOfInputs=numberofinputs;
        weight=new ArrayList<>(numberofinputs+1); // +1 для биаса
        input=new ArrayList<>(numberofinputs);
        activationFunction=iaf;
    }
    
    public void setNeuralLayer(NeuralLayer _neuralLayer){
        if(this.neuralLayer==null){
            this.neuralLayer=_neuralLayer;
        }
    }

    public void init(){
        init(new UniformInitialization(0.0,1.0));
    }
    
    public void init(WeightInitialization weightInit){ // Инициализация весов (случайные числа)
        if(numberOfInputs>0){
            for(int i=0;i<=numberOfInputs;i++){
                double newWeight = weightInit.Generate();
                try{
                    this.weight.set(i, newWeight);
                }
                catch(IndexOutOfBoundsException iobe){ // Предотвращение выхода за пределы
                    this.weight.add(newWeight);
                }
            }
        }
    }

    public void setInputs(double [] values){
        if(values.length==numberOfInputs){
            for(int i=0;i<numberOfInputs;i++){
                try{
                    input.set(i, values[i]);
                }
                catch(IndexOutOfBoundsException iobe){
                    input.add(values[i]);
                }
            }
        }
    }

    public void setInputs(ArrayList<Double> values){
        if(values.size()==numberOfInputs){
            input=values;
        }
    }

    public double getInput(int i){
        return input.get(i);
    }

    
    public Double getWeight(int i){
        return weight.get(i);
    }

    public void updateWeight(int i, double value){
        if(i>=0 && i<=numberOfInputs){
            weight.set(i, value);
        }
    }

    public int getNumberOfInputs(){
        return this.numberOfInputs;
    }

    public double getOutput(){
        return output;
    }

    public void calc(){            // Вычисление выходных значений
        outputBeforeActivation=0.0;
        if(numberOfInputs>0){
            if(input!=null && weight!=null){
                for(int i=0;i<=numberOfInputs;i++){
                    outputBeforeActivation+=(i==numberOfInputs?bias:input.get(i))*weight.get(i);
                }
            }
        }
        output=activationFunction.calc(outputBeforeActivation);
        if(neuralLayer.getNeuralMode()==NeuralNet.NeuralNetMode.TRAINING){
            firstDerivative=activationFunction.derivative(outputBeforeActivation);
        }
    }
    
    public Double derivative(double[] _input){
        Double _outputBeforeActivation=0.0;
        if(numberOfInputs>0){
            if(weight!=null){
                for(int i=0;i<=numberOfInputs;i++){
                    _outputBeforeActivation+=(i==numberOfInputs?bias:_input[i])*weight.get(i);
                }
            }
        }
        return activationFunction.derivative(_outputBeforeActivation);
    }
    
    public ArrayList<Double> derivativeBatch(ArrayList<ArrayList<Double>> _input){
        ArrayList<Double> result = new ArrayList<>();
        for(int i=0;i<_input.size();i++){
            result.add(0.0);
            Double _outputBeforeActivation=0.0;
            for(int j=0;j<numberOfInputs;j++){
                _outputBeforeActivation+=(j==numberOfInputs?bias:_input.get(i).get(j))*weight.get(j);
            }
            result.set(i,activationFunction.derivative(_outputBeforeActivation));
        }
        return result;
    }

    public void setActivationFunction(IActivationFunction iaf){
        this.activationFunction=iaf;
    }
    
}
