package neur.learn;

import neur.NeuralException;
import neur.NeuralNet;
import neur.data.NeuralDataSet;
import neur.init.UniformInitialization;
import neur.math.IActivationFunction;
import neur.math.Linear;
import neur.math.RandomNumberGenerator;
import neur.math.Sigmoid;

public class BackpropagationTest {
    public static void main(String[] args){
        RandomNumberGenerator.seed=850;
        
        int numberOfInputs=3;
        int numberOfOutputs=4;
        int[] numberOfHiddenNeurons={4,3,5};
        
        
        Linear outputAcFnc = new Linear(1.0);
        Sigmoid hl0Fnc = new Sigmoid(1.0);
        Sigmoid hl1Fnc = new Sigmoid(1.0);
        Sigmoid hl2Fnc = new Sigmoid(1.0);
        IActivationFunction[] hiddenAcFnc={hl0Fnc,hl1Fnc,hl2Fnc};
        System.out.println("Создание нейронной сети");
        NeuralNet nn = new NeuralNet(numberOfInputs, numberOfOutputs, numberOfHiddenNeurons, hiddenAcFnc, outputAcFnc,new UniformInitialization(-1.0,1.0));
        System.out.println("Нейросеть создана");
        nn.print();
        
        Double[][] _neuralDataSet = {
            {-1.0,-1.0,-1.0,-1.0,1.0,-3.0,1.0}
        ,   {-1.0,-1.0,1.0,1.0,-1.0,-1.0,-1.0}
        ,   {-1.0,1.0,-1.0,1.0,-1.0,-1.0,-1.0}
        ,   {-1.0,1.0,1.0,-1.0,-1.0,1.0,-3.0}
        ,   {1.0,-1.0,-1.0,1.0,-1.0,-1.0,3.0}
        ,   {1.0,-1.0,1.0,-1.0,-1.0,1.0,1.0}
        ,   {1.0,1.0,-1.0,-1.0,1.0,1.0,1.0}
        ,   {1.0,1.0,1.0,1.0,-1.0,3.0,-1.0}
        };
        
        int[] inputColumns = {0,1,2};
        int[] outputColumns = {3,4,5,6};
        
        NeuralDataSet neuralDataSet = new NeuralDataSet(_neuralDataSet,inputColumns,outputColumns);
        
        System.out.println("Набор данных задан");
        neuralDataSet.printInput();
        neuralDataSet.printTargetOutput();
        
        System.out.println("Получение начальных выходных значений нейросети:");
        
        Backpropagation backprop = new Backpropagation(nn,neuralDataSet, neur.learn.LearningAlgorithm.LearningMode.BATCH);
        backprop.setLearningRate(0.2);
        backprop.setMaxEpochs(20000);
        backprop.setGeneralErrorMeasurement(Backpropagation.ErrorMeasurement.SimpleError);
        backprop.setOverallErrorMeasurement(Backpropagation.ErrorMeasurement.MSE);
        backprop.setMinOverallError(0.0001);
        backprop.printTraining=true;
        backprop.setMomentumRate(0.7);
        
        try{
            backprop.forward();
            neuralDataSet.printNeuralOutput();
            
            backprop.train();
            System.out.print("Конец обучения, результат: ");
            if(backprop.getMinOverallError()>=backprop.getOverallGeneralError()){
                System.out.println("Успех!");
            }
            else{
                System.out.println("Неудача!");
            }
            System.out.println("Общая ошибка:" +String.valueOf(backprop.getOverallGeneralError()));
            System.out.println("Минимальная общая ошибка:" +String.valueOf(backprop.getMinOverallError()));
            System.out.println("Количество эпох:" +String.valueOf(backprop.getEpoch()));
            
            System.out.println("Целевые выходные значения:");
            neuralDataSet.printTargetOutput();
            
            System.out.println("Выходные значения после обучения нейросети:");
            backprop.forward();
            neuralDataSet.printNeuralOutput();
        }
        catch(NeuralException ne){
            System.out.println("Ошибка!");
        }

    }
}
