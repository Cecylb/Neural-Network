package neur.data;

import java.util.ArrayList;

public class NeuralOutputData {
    
    public int numberOfOutputs=0;
    public int numberOfRecords=0;
 
    public ArrayList<ArrayList<Double>> targetData;
    
    public ArrayList<ArrayList<Double>> neuralData;

    public NeuralOutputData(Double[][] _data){
        this.numberOfRecords=_data.length;
        this.targetData=new ArrayList<>();
        this.neuralData=new ArrayList<>();
        for(int i=0;i<numberOfRecords;i++){
            this.targetData.add(new ArrayList<Double>());
            this.neuralData.add(new ArrayList<Double>());
            if(this.numberOfOutputs==0){
                this.numberOfOutputs=_data[i].length;
            }
            for(int j=0;j<numberOfOutputs;j++){
                this.targetData.get(i).add(_data[i][j]);
                this.neuralData.get(i).add(null);
            }
        }        
    }
    
    public ArrayList<ArrayList<Double>> getTargetDataArrayList(){
        return this.targetData;
    }
    
    public ArrayList<ArrayList<Double>> getNeuralDataArrayList(){
        return this.neuralData;
    }

    public void setNeuralData(int i,double[] _data){
        for(int j=0;j<numberOfOutputs;j++){
            this.neuralData.get(i).set(j, _data[j]);
        }
    }
    
    public ArrayList<Double> getTargetRecordArrayList(int i){
        return this.targetData.get(i);
    }

    public ArrayList<Double> getRecordArrayList(int i){
        return this.neuralData.get(i);
    }
    
    public ArrayList<Double> getTargetColumnArrayList(int i){
        ArrayList<Double> result = new ArrayList<>();
        for(int j=0;j<numberOfRecords;j++){
            result.add(targetData.get(j).get(i));
        }
        return result;
    }
    
    public ArrayList<Double> getNeuralColumnArrayList(int i){
        ArrayList<Double> result = new ArrayList<>();
        for(int j=0;j<numberOfRecords;j++){
            result.add(neuralData.get(j).get(i));
        }
        return result;
    }
    
    public void printTarget(){
        for(int k=0;k<numberOfRecords;k++){
            System.out.print("Целевое выходное значение ["+String.valueOf(k)+"]={ ");
            for(int i=0;i<numberOfOutputs;i++){
                if(i==numberOfOutputs-1){
                    System.out.print(String.valueOf(this.targetData.get(k).get(i))+"}\n");
                }
                else{
                    System.out.print(String.valueOf(this.targetData.get(k).get(i))+"\t");
                }
            }
        }
    }
    
    public void printNeural(){
        for(int k=0;k<numberOfRecords;k++){
            System.out.print("Полученное выходное значение ["+String.valueOf(k)+"]={ ");
            for(int i=0;i<numberOfOutputs;i++){
                if(i==numberOfOutputs-1){
                    System.out.print(String.valueOf(this.neuralData.get(k).get(i))+"}\n");
                }
                else{
                    System.out.print(String.valueOf(this.neuralData.get(k).get(i))+"\t");
                }
            }
        }
    }
}
