package org.example;

import org.example.model.LVQ;

import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Hello world!
 *
 */
public class MainClass
{
    private static int[] epochs = {1};//,2,5};
    private static int[] neuronPerClass = /*{1,2,5,*/{10};
    private static double[] LR = {0.001};//,0.01,0.1};
    
    public static void main( String[] args )
    {
        
        StringBuilder sb = new StringBuilder();
        sb.append("epochs,neuronsPerClass,LR,testClass,result,guessIndex\n");
                
        for(int epoch:epochs){
            for(int npc:neuronPerClass){
                for(double rate:LR){
                    lvq(epoch,npc,rate,sb);
                    sb.append(",,,,,\n");
                }
            }
        }
        //DataLoader.Save(sb);
    }
    
    private static void lvq(int epochs, int neuronPerClass, double LR, StringBuilder sb){        
        //train data
        Map<Integer[],String> trainData = DataLoader.getTrainData();

        //test data
        Map<Integer[],String> testData = DataLoader.getTestData();

//        //epochs
//        int epochs = 1;
//        
//        //neuronPerClass
//        int neuronPerClass = 10;
//        
//        //epochs
//        double LR = 0.01;

        //create model
        LVQ model = new LVQ(trainData,epochs,neuronPerClass,LR);

        //train
        model.train();

        //classes
        Set<String> names = DataLoader.getTestNames();

        //testing
        for(String name:names){
            //current class vector set
            List<Integer[]> inputData = new LinkedList<>();

            //filter
            for(Integer[] vector:testData.keySet()){
                String ans = testData.get(vector);
                if(ans.equals(name)){
                    inputData.add(vector);
                }
            }

            //test
            model.test(inputData,name,sb);
        }        
    }
}
