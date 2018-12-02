
package neur.learn;

import neur.HiddenLayer;
import neur.NeuralException;
import neur.NeuralLayer;
import neur.NeuralNet;
import neur.Neuron;
import neur.OutputLayer;
import neur.data.NeuralDataSet;
import java.util.ArrayList;

public class Backpropagation extends LearningAlgorithm {

    public ArrayList<ArrayList<Double>> error;
    public ArrayList<Double> generalError;
    public ArrayList<Double> overallError;
    public double overallGeneralError;

    public double degreeGeneralError=2.0;
    public double degreeOverallError=0.0;

    public enum ErrorMeasurement {SimpleError, SquareError,NDegreeError,MSE}

    public ErrorMeasurement generalErrorMeasurement=Backpropagation.ErrorMeasurement.SquareError;
    public ErrorMeasurement overallErrorMeasurement=Backpropagation.ErrorMeasurement.MSE;

    protected int currentRecord=0;

    protected ArrayList<ArrayList<ArrayList<Double>>> newWeights;

    private double MomentumRate=0.7;
    
    public ArrayList<ArrayList<Double>> deltaNeuron;
    
    public ArrayList<ArrayList<ArrayList<Double>>> lastDeltaWeights;

   public Backpropagation(NeuralNet _neuralNet,NeuralDataSet _trainDataSet, Backpropagation.LearningMode _learningMode){
       this(_neuralNet,_trainDataSet);
       this.learningMode=_learningMode;
        initializeDeltaNeuron();
        initializeLastDeltaWeights();
    }
    
    private void initializeDeltaNeuron(){
        deltaNeuron=new ArrayList<>();
        int numberOfHiddenLayers =neuralNet.getNumberOfHiddenLayers();
        for(int l=0;l<=numberOfHiddenLayers;l++){
            int numberOfNeuronsInLayer;
            deltaNeuron.add(new ArrayList<Double>());
            if(l==numberOfHiddenLayers){
                numberOfNeuronsInLayer=neuralNet.getOutputLayer().getNumberOfNeuronsInLayer();
            }
            else{
                numberOfNeuronsInLayer=neuralNet.getHiddenLayer(l).getNumberOfNeuronsInLayer();
            }
            for(int j=0;j<numberOfNeuronsInLayer;j++){
                deltaNeuron.get(l).add(null);
            }
        }
    }
    
    private void initializeLastDeltaWeights(){
        this.lastDeltaWeights=new ArrayList<>();
        int numberOfHiddenLayers=this.neuralNet.getNumberOfHiddenLayers();
        for(int l=0;l<=numberOfHiddenLayers;l++){
            int numberOfNeuronsInLayer,numberOfInputsInNeuron;
            this.lastDeltaWeights.add(new ArrayList<ArrayList<Double>>());
            if(l<numberOfHiddenLayers){
                numberOfNeuronsInLayer=this.neuralNet.getHiddenLayer(l).getNumberOfNeuronsInLayer();
                for(int j=0;j<numberOfNeuronsInLayer;j++){
                    numberOfInputsInNeuron=this.neuralNet.getHiddenLayer(l).getNeuron(j).getNumberOfInputs();
                    this.lastDeltaWeights.get(l).add(new ArrayList<Double>());
                    for(int i=0;i<=numberOfInputsInNeuron;i++){
                        this.lastDeltaWeights.get(l).get(j).add(0.0);
                    }
                }
            }
            else{
                numberOfNeuronsInLayer=this.neuralNet.getOutputLayer().getNumberOfNeuronsInLayer();
                for(int j=0;j<numberOfNeuronsInLayer;j++){
                    numberOfInputsInNeuron=this.neuralNet.getOutputLayer().getNeuron(j).getNumberOfInputs();
                    this.lastDeltaWeights.get(l).add(new ArrayList<Double>());
                    for(int i=0;i<=numberOfInputsInNeuron;i++){
                        this.lastDeltaWeights.get(l).get(j).add(0.0);
                    }
                }
            }
        }
    }

    public Double calcDeltaWeight(int layer,int input,int neuron) {
        Double deltaWeight=1.0;
        switch(learningMode){
            case BATCH:
                deltaWeight*=LearningRate;
            case ONLINE:
                deltaWeight*=LearningRate;
        }
        NeuralLayer currLayer;
        Neuron currNeuron;
        double _deltaNeuron;
        if(layer==neuralNet.getNumberOfHiddenLayers()){ //output layer
            currLayer=neuralNet.getOutputLayer();
            currNeuron=currLayer.getNeuron(neuron);
            _deltaNeuron=error.get(currentRecord).get(neuron) *currNeuron.derivative(currLayer.getInputs());
        }
        else{ //hidden layer
            currLayer=neuralNet.getHiddenLayer(layer);
            currNeuron=currLayer.getNeuron(neuron);
            double sumDeltaNextLayer=0;
            NeuralLayer nextLayer=currLayer.getNextLayer();
            for(int k=0;k<nextLayer.getNumberOfNeuronsInLayer();k++){
                sumDeltaNextLayer+=nextLayer.getWeight(neuron, k) *deltaNeuron.get(layer+1).get(k);
            }
            _deltaNeuron=sumDeltaNextLayer*currNeuron.derivative(currLayer.getInputs());
            
        }
        
        deltaNeuron.get(layer).set(neuron, _deltaNeuron);
        deltaWeight*=_deltaNeuron;
        if(input<currNeuron.getNumberOfInputs()){
            deltaWeight*=currNeuron.getInput(input);
        }
        
        return deltaWeight;
    }
    
    
    @Override
    public void train() throws NeuralException{
        neuralNet.setNeuralNetMode(NeuralNet.NeuralNetMode.TRAINING);
        epoch=0;
        int k=0;
        currentRecord=0;
        forward();
        forward(k);
        if(printTraining){
            print();
        } 
        while(epoch<MaxEpochs && overallGeneralError>MinOverallError){
            backward();
            switch(learningMode){
                case BATCH:
                    if(k==trainingDataSet.numberOfRecords-1)
                        applyNewWeights();
                    break;
                case ONLINE:
                    applyNewWeights();
            }
            currentRecord=++k;
            if(k>=trainingDataSet.numberOfRecords){
                k=0;
                currentRecord=0;
                epoch++;
            }

            forward(k);
            if(printTraining && (learningMode==LearningMode.ONLINE || (k==0))){
                print();
            } 
            
        }
        neuralNet.setNeuralNetMode(NeuralNet.NeuralNetMode.RUN);
    }
    
    @Override
    public void forward(int i){
        neuralNet.setInputs(trainingDataSet.getInputRecord(i));
        neuralNet.calc();
        trainingDataSet.setNeuralOutput(i, neuralNet.getOutputs());
        generalError.set(i, generalError(trainingDataSet.getArrayTargetOutputRecord(i),trainingDataSet.getArrayNeuralOutputRecord(i)));
        for(int j=0;j<neuralNet.getNumberOfOutputs();j++){
            overallError.set(j, overallError(trainingDataSet.getIthTargetOutputArrayList(j), trainingDataSet.getIthNeuralOutputArrayList(j)));
            error.get(i).set(j,simpleError(trainingDataSet.getIthTargetOutputArrayList(j).get(i), trainingDataSet.getIthNeuralOutputArrayList(j).get(i)));
        }
        overallGeneralError=overallGeneralErrorArrayList(trainingDataSet.getArrayTargetOutputData(),trainingDataSet.getArrayNeuralOutputData());
    }

    @Override
    public void forward(){
        for(int i=0;i<trainingDataSet.numberOfRecords;i++){
            neuralNet.setInputs(trainingDataSet.getInputRecord(i));
            neuralNet.calc();
            trainingDataSet.setNeuralOutput(i, neuralNet.getOutputs());
            generalError.set(i, generalError(trainingDataSet.getArrayTargetOutputRecord(i),trainingDataSet.getArrayNeuralOutputRecord(i)));
            for(int j=0;j<neuralNet.getNumberOfOutputs();j++){
                error.get(i).set(j,simpleError(trainingDataSet.getArrayTargetOutputRecord(i).get(j), trainingDataSet.getArrayNeuralOutputRecord(i).get(j)));
            }
        }
        for(int j=0;j<neuralNet.getNumberOfOutputs();j++){
            overallError.set(j, overallError(trainingDataSet.getIthTargetOutputArrayList(j), trainingDataSet.getIthNeuralOutputArrayList(j)));
        }
        overallGeneralError=overallGeneralErrorArrayList(trainingDataSet.getArrayTargetOutputData(),trainingDataSet.getArrayNeuralOutputData());
    }

    public void backward(){
        int numberOfLayers=neuralNet.getNumberOfHiddenLayers();
        for(int l=numberOfLayers;l>=0;l--){
            int numberOfNeuronsInLayer=deltaNeuron.get(l).size();
            for(int j=0;j<numberOfNeuronsInLayer;j++){
                for(int i=0;i<newWeights.get(l).get(j).size();i++){
                    double currNewWeight = this.newWeights.get(l).get(j).get(i);
                    if(currNewWeight==0.0 && epoch==0.0)
                        if(l==numberOfLayers)
                            currNewWeight=neuralNet.getOutputLayer().getWeight(i, j);
                        else
                            currNewWeight=neuralNet.getHiddenLayer(l).getWeight(i, j);
                    double deltaWeight=calcDeltaWeight(l, i, j);
                    newWeights.get(l).get(j).set(i,currNewWeight+deltaWeight);
                }
            }
        }
    }

    @Override
    public void applyNewWeights(){
        int numberOfHiddenLayers=this.neuralNet.getNumberOfHiddenLayers();
        for(int l=0;l<=numberOfHiddenLayers;l++){
            int numberOfNeuronsInLayer,numberOfInputsInNeuron;
            if(l<numberOfHiddenLayers){
                HiddenLayer hl = this.neuralNet.getHiddenLayer(l);
                numberOfNeuronsInLayer=hl.getNumberOfNeuronsInLayer();
                for(int j=0;j<numberOfNeuronsInLayer;j++){
                    numberOfInputsInNeuron=hl.getNeuron(j).getNumberOfInputs();
                    for(int i=0;i<=numberOfInputsInNeuron;i++){
                        Double lastDeltaWeight=lastDeltaWeights.get(l).get(j).get(i);
                        double momentum=MomentumRate*lastDeltaWeight;
                        double newWeight=this.newWeights.get(l).get(j).get(i) -momentum;
                        this.newWeights.get(l).get(j).set(i,newWeight);
                        Neuron n=hl.getNeuron(j);
                        double deltaWeight=(newWeight-n.getWeight(i));
                        lastDeltaWeights.get(l).get(j).set(i,(double)deltaWeight);
                        hl.getNeuron(j).updateWeight(i, newWeight);
                    }
                }
            }
            else{
                OutputLayer ol = this.neuralNet.getOutputLayer();
                numberOfNeuronsInLayer=ol.getNumberOfNeuronsInLayer();
                for(int j=0;j<numberOfNeuronsInLayer;j++){
                    numberOfInputsInNeuron=ol.getNeuron(j).getNumberOfInputs();
                    
                    for(int i=0;i<=numberOfInputsInNeuron;i++){
                        Double lastDeltaWeight=lastDeltaWeights.get(l).get(j).get(i);
                        double momentum=MomentumRate*lastDeltaWeight;
                        Neuron n=ol.getNeuron(j);
                        double newWeight=this.newWeights.get(l).get(j).get(i) + momentum;
                        this.newWeights.get(l).get(j).set(i,newWeight);
                        double deltaWeight=(newWeight-n.getWeight(i));
                        lastDeltaWeights.get(l).get(j).set(i,deltaWeight);
                        ol.getNeuron(j).updateWeight(i, newWeight);
                    }
                }
            }
        }        
    }
    
    public void setMomentumRate(double _momentumRate){
        this.MomentumRate=_momentumRate;
    }

    public Backpropagation(NeuralNet _neuralNet){
        this.learningParadigm=LearningParadigm.SUPERVISED;
        this.neuralNet=_neuralNet;
        this.newWeights=new ArrayList<>();
        int numberOfHiddenLayers=this.neuralNet.getNumberOfHiddenLayers();
        for(int l=0;l<=numberOfHiddenLayers;l++){
            int numberOfNeuronsInLayer,numberOfInputsInNeuron;
            this.newWeights.add(new ArrayList<ArrayList<Double>>());
            if(l<numberOfHiddenLayers){
                numberOfNeuronsInLayer=this.neuralNet.getHiddenLayer(l).getNumberOfNeuronsInLayer();
                for(int j=0;j<numberOfNeuronsInLayer;j++){
                    numberOfInputsInNeuron=this.neuralNet.getHiddenLayer(l).getNeuron(j).getNumberOfInputs();
                    this.newWeights.get(l).add(new ArrayList<Double>());
                    for(int i=0;i<=numberOfInputsInNeuron;i++){
                        this.newWeights.get(l).get(j).add(0.0);
                    }
                }
            }
            else{
                numberOfNeuronsInLayer=this.neuralNet.getOutputLayer().getNumberOfNeuronsInLayer();
                for(int j=0;j<numberOfNeuronsInLayer;j++){
                    numberOfInputsInNeuron=this.neuralNet.getOutputLayer().getNeuron(j).getNumberOfInputs();
                    this.newWeights.get(l).add(new ArrayList<Double>());
                    for(int i=0;i<=numberOfInputsInNeuron;i++){
                        this.newWeights.get(l).get(j).add(0.0);
                    }
                }
            }
        }
    }

    public Backpropagation(NeuralNet _neuralNet,NeuralDataSet _trainDataSet){
        this(_neuralNet);
        this.trainingDataSet=_trainDataSet;
        this.generalError=new ArrayList<>();
        this.error=new ArrayList<>();
        this.overallError=new ArrayList<>();
        for(int i=0;i<_trainDataSet.numberOfRecords;i++){
            this.generalError.add(null);
            this.error.add(new ArrayList<Double>());
            for(int j=0;j<_neuralNet.getNumberOfOutputs();j++){
                if(i==0){
                    this.overallError.add(null);
                }
                this.error.get(i).add(null);
            }
        }
    }

    public void setGeneralErrorMeasurement(ErrorMeasurement _errorMeasurement){
        switch(_errorMeasurement){
            case SimpleError:
                this.degreeGeneralError=1;
                break;
            case SquareError:
            case MSE:
                this.degreeGeneralError=2;
        }
        this.generalErrorMeasurement=_errorMeasurement;
    }

    public void setOverallErrorMeasurement(ErrorMeasurement _errorMeasurement){
        switch(_errorMeasurement){
            case SimpleError:
                this.degreeOverallError=1;
                break;
            case SquareError:
            case MSE:
                this.degreeOverallError=2;
        }
        this.overallErrorMeasurement=_errorMeasurement;
    }

    public Double overallGeneralErrorArrayList(ArrayList<ArrayList<Double>> YT,ArrayList<ArrayList<Double>> Y){
        int N=YT.size();
        int Ny=YT.get(0).size();
        Double result=0.0;
        for(int i=0;i<N;i++){
            Double resultY = 0.0;
            for(int j=0;j<Ny;j++){
                resultY+=Math.pow(YT.get(i).get(j)-Y.get(i).get(j), degreeGeneralError);
            }
            if(generalErrorMeasurement==ErrorMeasurement.MSE)
                result+=Math.pow((1.0/Ny)*resultY,degreeOverallError);
            else
                result+=Math.pow((1.0/degreeGeneralError)*resultY,degreeOverallError);
        }
        if(overallErrorMeasurement==ErrorMeasurement.MSE)
            result*=(1.0/N);
        else
            result*=(1.0/degreeOverallError);
        return result;
    }

    public Double generalError(ArrayList<Double> YT,ArrayList<Double> Y){
        int Ny=YT.size();
        Double result=0.0;
        for(int i=0;i<Ny;i++){
            result+=Math.pow(YT.get(i)-Y.get(i), degreeGeneralError);
        }
        if(generalErrorMeasurement==ErrorMeasurement.MSE)
            result*=(1.0/Ny);
        else
            result*=(1.0/degreeGeneralError);
        return result;
    }

    public Double overallError(ArrayList<Double> YT,ArrayList<Double> Y){
        int N=YT.size();
        Double result=0.0;
        for(int i=0;i<N;i++){
            result+=Math.pow(YT.get(i)-Y.get(i), degreeOverallError);
        }
        if(overallErrorMeasurement==ErrorMeasurement.MSE)
            result*=(1.0/N);
        else
            result*=(1.0/degreeOverallError);
        return result;
    }


    public Double simpleError(Double YT,Double Y){
        return YT-Y;
    }

    @Override
    public void print(){
        if(learningMode==LearningMode.ONLINE)
            System.out.println("Эпоха = "+String.valueOf(epoch)+"; Запись = " +String.valueOf(currentRecord)+"; Общая ошибка = " +String.valueOf(overallGeneralError));
        else
            System.out.println("Эпоха = "+String.valueOf(epoch) +"; Общая ошибка = "+String.valueOf(overallGeneralError));
    }

    public Double getOverallGeneralError(){
        return overallGeneralError;
    }
}
