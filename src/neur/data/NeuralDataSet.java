package neur.data;

import java.util.ArrayList;

public class NeuralDataSet {           // Обработка данных процесса обучения
    public NeuralInputData inputData;
    public NeuralOutputData outputData;
    
    public int numberOfInputs;
    public int numberOfOutputs;
    
    public int numberOfRecords;
    
    public NeuralDataSet(Double[][] _data,int[] inputColumns,int[] outputColumns){
        numberOfInputs=inputColumns.length;
        numberOfOutputs=outputColumns.length;
        numberOfRecords=_data.length;
        Double[][] _inputData=new Double[numberOfRecords][numberOfInputs];
        Double[][] _outputData=new Double[numberOfRecords][numberOfOutputs];
        for(int i=0;i<numberOfInputs;i++){
            for(int j=0;j<numberOfRecords;j++){
                _inputData[j][i]=_data[j][inputColumns[i]];
            }
        }
        for(int i=0;i<numberOfOutputs;i++){
            for(int j=0;j<numberOfRecords;j++){
                _outputData[j][i]=_data[j][outputColumns[i]];
            }
        }
        inputData=new NeuralInputData(_inputData);
        outputData=new NeuralOutputData(_outputData);
    }
    
    public ArrayList<ArrayList<Double>> getArrayInputData(){
        return inputData.data;
    }
    
    public ArrayList<ArrayList<Double>> getArrayTargetOutputData(){
        return outputData.getTargetDataArrayList();
    }
    
    public ArrayList<ArrayList<Double>> getArrayNeuralOutputData(){
        return outputData.getNeuralDataArrayList();
    }
    
    public ArrayList<Double> getArrayInputRecord(int i){
        return inputData.getRecordArrayList(i);
    }
    
    public double[] getInputRecord(int i){
        return inputData.getRecord(i);
    }
    
    public ArrayList<Double> getArrayTargetOutputRecord(int i){
        return outputData.getTargetRecordArrayList(i);
    }
    
    public ArrayList<Double> getArrayNeuralOutputRecord(int i){
        return outputData.getRecordArrayList(i);
    }
    
    public void setNeuralOutput(int i,double[] _neuralData){
        this.outputData.setNeuralData(i, _neuralData);
    }
    
    public ArrayList<Double> getIthInputArrayList(int i){
        return this.inputData.getColumnDataArrayList(i);
    }

    public ArrayList<Double> getIthTargetOutputArrayList(int i){
        return this.outputData.getTargetColumnArrayList(i);
    }
    
    public ArrayList<Double> getIthNeuralOutputArrayList(int i){
        return this.outputData.getNeuralColumnArrayList(i);
    }
    
    public void printInput(){
        this.inputData.print();
    }
    
    public void printTargetOutput(){
        this.outputData.printTarget();
    }
    
    public void printNeuralOutput(){
        this.outputData.printNeural();
    }

}
