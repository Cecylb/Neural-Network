package neur;

import neur.math.IActivationFunction;

public class OutputLayer extends NeuralLayer {

    public OutputLayer(NeuralNet _neuralNet,int numberofneurons,IActivationFunction iaf,int numberofinputs){
        super(_neuralNet,numberofneurons,iaf);
        numberOfInputs=numberofinputs;
        nextLayer=null;
        init(_neuralNet.getWeightInitialization());
    }

    @Override
    public void setNextLayer(NeuralLayer layer){
        nextLayer=null;
    }

    @Override
    public void setPreviousLayer(NeuralLayer layer){
        previousLayer=layer;
        if(layer.nextLayer!=this)
            layer.setNextLayer(this);
    }
    
}
