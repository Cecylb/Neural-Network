package neur.data;

import java.util.ArrayList;

public class NeuralInputData {
    
    public int numberOfInputs=0;
    public int numberOfRecords=0;
 
    public ArrayList<ArrayList<Double>> data;

    public NeuralInputData(Double[][] _data){
        this.numberOfRecords=_data.length;
        this.data=new ArrayList<>();
        for(int i=0;i<numberOfRecords;i++){
            this.data.add(new ArrayList<Double>());
            if(this.numberOfInputs==0){
                this.numberOfInputs=_data[i].length;
            }
            for(int j=0;j<numberOfInputs;j++){
                this.data.get(i).add(_data[i][j]);
            }
        }
    }
    
    public ArrayList<Double> getRecordArrayList(int i){
        return this.data.get(i);
    }
    
    public double[] getRecord(int i){
        double[] result=new double[numberOfInputs];
        for(int j=0;j<numberOfInputs;j++){
            result[j]=this.data.get(i).get(j);
        }
        return result;
    }
    
    public ArrayList<Double> getColumnDataArrayList(int i){
        ArrayList<Double> result=new ArrayList<>();
        for(int j=0;j<numberOfRecords;j++){
            result.add(data.get(j).get(i));
        }
        return result;
    }
    
    public void print(){
        System.out.println("Входные значения:");
        for(int k=0;k<numberOfRecords;k++){
            System.out.print("Вход ["+String.valueOf(k)+"]={ ");
            for(int i=0;i<numberOfInputs;i++){
                if(i==numberOfInputs-1){
                    System.out.print(String.valueOf(this.data.get(k).get(i))+"}\n");
                }
                else{
                    System.out.print(String.valueOf(this.data.get(k).get(i))+"\t");
                }
            }
        }
    }
}
