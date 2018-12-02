package neur;

import neur.init.UniformInitialization;
import neur.init.WeightInitialization;
import neur.math.IActivationFunction;
import java.util.ArrayList;

public class NeuralNet {

    private InputLayer inputLayer; // Входной слой
    private ArrayList<HiddenLayer> hiddenLayer; // Массив внутренних слоёв
    private OutputLayer outputLayer; // Выходной слой
    private int numberOfHiddenLayers; // Число внутренних слоёв
    private int numberOfInputs; // Число входов
    private int numberOfOutputs; // Число выходов
    private ArrayList<Double> input; // Массив входов
    private ArrayList<Double> output; // Массив выходов
    private WeightInitialization weightInitialization = new UniformInitialization(0.0,1.0);
    private int[] neuronsInHiddenLayers;
    private int[] indexesWeightPerLayer;
    public enum NeuralNetMode { BUILD, TRAINING, RUN };
    
    private NeuralNetMode neuralNetMode = NeuralNetMode.BUILD;

    public NeuralNet(int numberofinputs,int numberofoutputs, int [] numberofhiddenneurons,IActivationFunction[] hiddenAcFnc, IActivationFunction outputAcFnc, WeightInitialization _weightInitialization){
        weightInitialization=_weightInitialization;
        numberOfHiddenLayers=numberofhiddenneurons.length;
        neuronsInHiddenLayers = new int[numberOfHiddenLayers+1];
        indexesWeightPerLayer = new int[numberOfHiddenLayers+2];  
        for(int i=0;i<=numberOfHiddenLayers;i++){
            if(i==numberOfHiddenLayers){
                neuronsInHiddenLayers[i]=numberofoutputs;
            }
            else{
                neuronsInHiddenLayers[i]=numberofhiddenneurons[i];
            }
            if(i==0){
                indexesWeightPerLayer[i]=0;
            }
            else{
                indexesWeightPerLayer[i]=indexesWeightPerLayer[i-1] + (neuronsInHiddenLayers[i-1]* ((i==1?numberofinputs:neuronsInHiddenLayers[i-2]) +1));
            }
        }
        if(numberOfHiddenLayers>0){
            indexesWeightPerLayer[numberOfHiddenLayers+1]= indexesWeightPerLayer[numberOfHiddenLayers] + neuronsInHiddenLayers[numberOfHiddenLayers] *(neuronsInHiddenLayers[numberOfHiddenLayers-1]+1);
        }
        else{
            indexesWeightPerLayer[numberOfHiddenLayers+1]= indexesWeightPerLayer[numberOfHiddenLayers] + neuronsInHiddenLayers[numberOfHiddenLayers] *(numberOfInputs+1);
        }
        numberOfInputs=numberofinputs;
        numberOfOutputs=numberofoutputs;
        if(numberOfHiddenLayers==hiddenAcFnc.length){
            input=new ArrayList<>(numberofinputs);
            inputLayer=new InputLayer(this,numberofinputs);
            if(numberOfHiddenLayers>0){
                hiddenLayer=new ArrayList<>(numberOfHiddenLayers);
            }
            for(int i=0;i<numberOfHiddenLayers;i++){
                if(i==0){
                    try{
                        hiddenLayer.set(i,new HiddenLayer(this,numberofhiddenneurons[i], hiddenAcFnc[i], inputLayer.getNumberOfNeuronsInLayer()));
                    }
                    catch(IndexOutOfBoundsException iobe){
                        hiddenLayer.add(new HiddenLayer(this,numberofhiddenneurons[i], hiddenAcFnc[i], inputLayer.getNumberOfNeuronsInLayer()));
                    }
                    inputLayer.setNextLayer(hiddenLayer.get(i));
                }
                else{
                    try{
                        hiddenLayer.set(i, new HiddenLayer(this,numberofhiddenneurons[i], hiddenAcFnc[i],hiddenLayer.get(i-1).getNumberOfNeuronsInLayer()));
                    }
                    catch(IndexOutOfBoundsException iobe){
                        hiddenLayer.add(new HiddenLayer(this,numberofhiddenneurons[i], hiddenAcFnc[i],hiddenLayer.get(i-1).getNumberOfNeuronsInLayer()));
                    }
                    hiddenLayer.get(i-1).setNextLayer(hiddenLayer.get(i));
                }
            }
            if(numberOfHiddenLayers>0){
                outputLayer=new OutputLayer(this,numberofoutputs,outputAcFnc, hiddenLayer.get(numberOfHiddenLayers-1).getNumberOfNeuronsInLayer());
                hiddenLayer.get(numberOfHiddenLayers-1).setNextLayer(outputLayer);
            }
            else{
                outputLayer=new OutputLayer(this,numberofoutputs, outputAcFnc, numberofinputs);
                inputLayer.setNextLayer(outputLayer);
            }
        }
        setNeuralNetMode(NeuralNetMode.RUN);
    }
    
    public NeuralNet(int numberofinputs,int numberofoutputs, int [] numberofhiddenneurons,IActivationFunction[] hiddenAcFnc, IActivationFunction outputAcFnc){
        this(numberofinputs,numberofoutputs,numberofhiddenneurons,hiddenAcFnc,outputAcFnc,new UniformInitialization(0.0,1.0));
    }
    
    public NeuralNet(int numberofinputs,int numberofoutputs, IActivationFunction outputAcFnc){
        this(numberofinputs,numberofoutputs,new int[0],new IActivationFunction[0],outputAcFnc);
    }

    public void setInputs(ArrayList<Double> inputs){
        if(inputs.size()==numberOfInputs){
            this.input=inputs;
        }
    }

    public void setInputs(double[] inputs){
        if(inputs.length==numberOfInputs){
            for(int i=0;i<numberOfInputs;i++){
                try{
                    input.set(i, inputs[i]);
                }
                catch(IndexOutOfBoundsException iobe){
                    input.add(inputs[i]);
                }
            }
        }
    }
    
    public Double getInput(int i){
        return input.get(i);
    }
    
    public double[] getInputs(){
        double[] result=new double[numberOfInputs];
        for(int i=0;i<numberOfInputs;i++){
            result[i]=input.get(i);
        }
        return result;
    }

    public void calc(){
        inputLayer.setInputs(input);
        inputLayer.calc();
        if(numberOfHiddenLayers>0){
            for(int i=0;i<numberOfHiddenLayers;i++){
                HiddenLayer hl = hiddenLayer.get(i);
                hl.setInputs(hl.getPreviousLayer().getOutputs());
                hl.calc();
            }
        }
        outputLayer.setInputs(outputLayer.getPreviousLayer().getOutputs());
        outputLayer.calc();
        this.output=outputLayer.getOutputs();
    }

    public double[] getOutputs(){
        double[] _outputs = new double[numberOfOutputs];
        for(int i=0;i<numberOfOutputs;i++){
            _outputs[i]=output.get(i);
        }
        return _outputs;
    }

    public void print(){
        System.out.println("Нейросеть: "+this.toString());
        System.out.println("\tЧисло входов:"+String.valueOf(this.numberOfInputs));
        System.out.println("\tЧисло выходов:"+String.valueOf(this.numberOfOutputs));
        System.out.println("\tЧисло внутренних слоёв: "+String.valueOf(numberOfHiddenLayers));
        for(int i=0;i<numberOfHiddenLayers;i++){
            System.out.println("\t\tВнутренний слой "+ String.valueOf(i)+" содержит: "+ String.valueOf(this.hiddenLayer.get(i).numberOfNeuronsInLayer)+" нейронов");
        }
        
    }

    public int getNumberOfHiddenLayers(){
        return numberOfHiddenLayers;
    }
    
    public int getNumberOfInputs(){
        return numberOfInputs;
    }
    
    public int getNumberOfOutputs(){
        return numberOfOutputs;
    }
    
    public HiddenLayer getHiddenLayer(int i){
        return hiddenLayer.get(i);
    }
    
    public OutputLayer getOutputLayer(){
        return outputLayer;
    }
    
    public WeightInitialization getWeightInitialization(){
        return weightInitialization;
    }
    
    public void setNeuralNetMode(NeuralNet.NeuralNetMode _neuralNetMode){
        this.neuralNetMode=_neuralNetMode;
    }
    
    public NeuralNetMode getNeuralNetMode(){
        return this.neuralNetMode;
    }
    
}
