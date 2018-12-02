package neur;

import neur.math.IActivationFunction;

public class HiddenLayer extends NeuralLayer {
    public HiddenLayer(NeuralNet _neuralNet, int numberofneurons,IActivationFunction iaf,int numberofinputs){
        super(_neuralNet,numberofneurons,iaf);
        numberOfInputs=numberofinputs;
        this.init(_neuralNet.getWeightInitialization());
    }
    @Override
    public void setPreviousLayer(NeuralLayer previous){
        this.previousLayer=previous;
        if(previous.nextLayer!=this)
            previous.setNextLayer(this);
    }
    @Override
    public void setNextLayer(NeuralLayer next){
        nextLayer=next;
        if(next.previousLayer!=this)
            next.setPreviousLayer(this);
    }
    
}
