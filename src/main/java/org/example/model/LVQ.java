package org.example.model;

import org.example.DataLoader;

import java.util.*;

public class LVQ {

    private List<Neuron> inputLayer;
    private List<Neuron> competitiveLayer;
    private List<Neuron> outputLayer;
    private List<Connection> nodesConnections;

    private Map<Integer[], String> trainData;

    private int epochs;
    private int neuronsPerClass;
    private double startLearningRate;
    private double learningRate;
    
    private List<String> results;

    public LVQ(Map<Integer[], String> trainData, int epochs, int neuronsPerClass, double startLearningRate){
        results = new LinkedList<>();
        this.neuronsPerClass = neuronsPerClass;
        //data
        this.trainData = trainData;

        //epochs count
        this.epochs = epochs;

        //LR
        this.startLearningRate = startLearningRate;

        //input vector length
        Integer[] someVector = (Integer[])trainData.keySet().toArray()[0];
        int inputVectorLength = someVector.length;

        //input layer
        inputLayer = new LinkedList<>();
        for (int i = 0; i < inputVectorLength; i++) {
            inputLayer.add(new Neuron());
        }

        Set<String> names = DataLoader.getTestNames();
        int numClasses = names.size();
        //hidden layer
        competitiveLayer = new LinkedList<>();
        for (int i = 0; i < numClasses; i++) {
            String name = (String) names.toArray()[i];
            for (int j = 0; j < neuronsPerClass; j++) {
                competitiveLayer.add(new Neuron(name));
            }
        }

//        //output layer
//        outputLayer = new LinkedList<>();
//        Set<String> names = DataLoader.getTestNames();
//        for (int i = 0; i < numClasses; i++) {
//            String name = (String) names.toArray()[i];
//            outputLayer.add(new Neuron(name));
//        }

        //link input and hidden layers
        nodesConnections = new LinkedList<>();
        
        List<String> namesList = new ArrayList<>(names);
        int[][][] firstWeights = new int[numClasses][neuronsPerClass][3];
        for(Integer[] line:trainData.keySet()){
            String ans = trainData.get(line);
            int index = namesList.indexOf(ans);
            
            //not filled
            if(firstWeights[index][neuronsPerClass-1][0] == 0){
                for(int i=0; i < neuronsPerClass; i++){
                    if(firstWeights[index][i][0]==0){
                        firstWeights[index][i][0] = line[0];
                        firstWeights[index][i][1] = line[1];
                        firstWeights[index][i][2] = line[2];
                        break;
                    }
                }
            }
        }
               
        for(Neuron hidden:competitiveLayer){           
            int index = Math.floorDiv(competitiveLayer.indexOf(hidden), neuronsPerClass);
            int neuronNumber = Math.floorMod(competitiveLayer.indexOf(hidden), neuronsPerClass);
            int[] randomVectorWithSameLabel = firstWeights[index][neuronNumber];
            
            for (Neuron input:inputLayer) {
                //weights
                nodesConnections.add(new Connection(input,hidden,randomVectorWithSameLabel[inputLayer.indexOf(input)]));
            }
        }

//        //link hidden and output layers
//        for (Neuron hidden:competitiveLayer) {
//            for(Neuron output:outputLayer){
//                nodesConnections.add(new Connection(hidden,output));
//            }
//        }
    }

    /**
     * model training
     */
    public void train(){

        //epoch iter
        for (int epoch = 0; epoch < epochs; epoch++) {
            //dataset
            List<Integer[]> curEpochData = new LinkedList<>();
            curEpochData.addAll(trainData.keySet());

            //for all train vectors
            for (int i = 0; i < trainData.keySet().size(); i++) {
                //print
                System.out.println(epoch+"/"+epochs+"   "+i+"/"+trainData.keySet().size());

                //calc LR
                calcLearningRate(i,epoch);

                //choose random vector
                int randomIndex = (int)(Math.random() * curEpochData.size());
                Integer[] inputVector = curEpochData.get(randomIndex);
                curEpochData.remove(inputVector);
                String className = trainData.get(inputVector);

                //best matching hidden unit
                Neuron BMU = getBMU(inputVector);

                String ans = trainData.get(inputVector);

                //изменить веса
                adjustHiddenWeights(BMU, inputVector, ans);

//                //activation
//                double[] hiddenOutput = hiddenActivation(BMU);
//
//                //define class
//                Neuron outputWinner = classify(hiddenOutput);
//
//                //weights
//                adjustOutputWeights(outputWinner,className,hiddenOutput);
            }
        }
        //debug point
        System.out.println("train complete");
    }

    private double[] hiddenActivation(Neuron BMU){
        double[] hiddenOutput = new double[competitiveLayer.size()];
        //v1
        //hidden layer output [0,0,...,1,...,0]
                int competitiveWinnerIndex = competitiveLayer.indexOf(BMU);
                hiddenOutput[competitiveWinnerIndex] = 1.0;
        //v2
//        double totalSum = 0;
//        for(Neuron hidden:competitiveLayer){
//            List<Connection> connections = getConnectionsWithEndpoint(hidden);
//            double sum = 0;
//            for(Connection c:connections){
//                sum += c.getWeight();
//            }
//            hiddenOutput[competitiveLayer.indexOf(hidden)] = sum;
//            totalSum+=sum;
//        }
//        for (int i = 0; i < hiddenOutput.length; i++) {
//            hiddenOutput[i] = hiddenOutput[i] / totalSum;
//            //System.out.println(hiddenOutput[i]);
//        }
        return hiddenOutput;
    }

    private void adjustHiddenWeights(Neuron BMU, Integer[] inputVector, String ans){
        //get winner weights
        List<Connection> winnerWeights = getConnectionsWithEndpoint(BMU);

        //move closer to input winner weights
        for (int i = 0; i < inputVector.length; i++) {
            double newWeightValue = winnerWeights.get(i).getWeight();
            if(ans.equals(BMU.getClassDescription())){
                newWeightValue += learningRate * (inputVector[i] - winnerWeights.get(i).getWeight());
            }
            else{
                //newWeightValue -= learningRate * (inputVector[i] - winnerWeights.get(i).getWeight());
            }
            //0..63 bounds
            //if(newWeightValue>63) newWeightValue = 63;
            //if(newWeightValue<0) newWeightValue = 0;
            winnerWeights.get(i).setWeight(newWeightValue);
        }
    }

    private Neuron classify(double[] hiddenOutput){
        Neuron winner = null;
        double mind= Double.POSITIVE_INFINITY;

        for (Neuron output:outputLayer) {
            //get weights
            List<Connection> curConnections = getConnectionsWithEndpoint(output);

            //calc sum
            double d = 0;
            for (int i = 0; i < hiddenOutput.length; i++) {
                d+=hiddenOutput[i] - curConnections.get(i).getWeight();
            }

            //new winner
            if(d < mind){
                mind = d;
                winner = output;
            }
        }

        return winner;
    }

    private void adjustOutputWeights(Neuron output, String ans, double[] hiddenOutput){
        List<Connection> winnerWeights = getConnectionsWithEndpoint(output);

        //same class
        if(output.getClassDescription().equals(ans)){
            for (int i = 0; i < hiddenOutput.length; i++) {
                if(Math.abs(hiddenOutput[i]) < 0.1) continue;
                double newWeightValue = winnerWeights.get(i).getWeight();
                newWeightValue += learningRate * (hiddenOutput[i] - winnerWeights.get(i).getWeight());
                winnerWeights.get(i).setWeight(newWeightValue);
            }
        }
        //wrong class
        else {
//            for (int i = 0; i < hiddenOutput.length; i++) {
//                if(Math.abs(hiddenOutput[i]) < 0.1) continue;
//                double newWeightValue = winnerWeights.get(i).getWeight();
//                newWeightValue -= learningRate * (hiddenOutput[i] - winnerWeights.get(i).getWeight());
//                winnerWeights.get(i).setWeight(newWeightValue);
//            }
        }
    }

    /**
     *
     * @param inputVector
     * @return
     */
    private Neuron getBMU(Integer[] inputVector){
        Neuron BMU = null;
        double curMinDistance = Double.POSITIVE_INFINITY;

        for (Neuron hidden:competitiveLayer) {
            List<Connection> endpointsConnections = getConnectionsWithEndpoint(hidden);
            double curDistance = 0;

            for (Neuron input:inputLayer) {
                for (Connection c:endpointsConnections) {
                    if(c.getStartNode().equals(input)){
                        double w = c.getWeight();
                        double v = inputVector[inputLayer.indexOf(input)];
                        curDistance += Math.pow(v-w,2);
                        break;
                    }
                }
            }
            curDistance = Math.sqrt(curDistance);
            if(curDistance < curMinDistance){
                curMinDistance = curDistance;
                BMU = hidden;
            }
        }
        return BMU;
    }

    private List<Connection> getConnectionsWithEndpoint(Neuron endpoint){
        List<Connection> connections = new LinkedList<>();
        for (Connection c:nodesConnections) {
            if(c.getEndNode().equals(endpoint)){
                connections.add(c);
            }
        }
        return connections;
    }

    private void calcLearningRate(int iteration, int curEpoch) {
        int dataLength = trainData.keySet().size();
        double expValue = -(double)(iteration + curEpoch * dataLength) / (double)(dataLength * epochs);
        learningRate = startLearningRate * Math.exp(expValue);
    }

    public void test(List<Integer[]> testData, String ans, StringBuilder sb){
        //class count
        Map<String,Integer> count = new HashMap<>();
        for(Neuron n:competitiveLayer){
            if(!count.containsKey(n.getClassDescription()))
                count.put(n.getClassDescription(),0);
        }

        //for each line in file
        for (Integer[] vector:testData) {

            //get winner class
            Neuron winner = getBMU(vector);

            //count
            for (String name:count.keySet()) {
                if(name.equals(winner.getClassDescription())){
                    int v = count.get(name) + 1;
                    count.remove(name);
                    count.put(name,v);
                    break;
                }
            }

        }
        int linesInFile = testData.size();
        System.out.println("results for "+linesInFile+" lines in "+ans);
        sb.append(epochs+","+neuronsPerClass+","+startLearningRate+","+ans+",");
        boolean append = false;

        int notNullCount = 0;
        for (String name:count.keySet()) {
            if(count.get(name)!=0){
                notNullCount++;
            }
        }

        // max 3
        for (int i = 0; i < notNullCount; i++) {
            int curMax = 0;
            String maxName = "";
            for (String name:count.keySet()) {
                int v = count.get(name);
                if(v > curMax){
                    curMax = v;
                    maxName = name;
                }
            }
            count.remove(maxName);
            if(maxName.equals(ans)){
                System.out.print(i+") ===> ");
                //results.add(i+") ===> "+maxName+" "+(double)curMax/(double)linesInFile);
                sb.append((double)curMax/(double)linesInFile+","+i+"\n");
                append = true;
            }
            else{
                System.out.print(i+") ");
                //results.add(i+") "+maxName+" "+(double)curMax/(double)linesInFile);
            }
            System.out.println(maxName+" "+(double)curMax/(double)linesInFile);
        }
        if(!append){
            sb.append("-1,-1\n");
        }
        
    }

    private Neuron getWinnerClass(Integer[] inputVector){
        //best matching hidden unit
        Neuron BMU = getBMU(inputVector);

//        //hidden layer output [0,0,...,1,...,0]
//        double[] hiddenOutput = new double[competitiveLayer.size()];
//        int competitiveWinnerIndex = competitiveLayer.indexOf(BMU);
//        hiddenOutput[competitiveWinnerIndex] = 1.0;
//
//        //define class
//        Neuron outputWinner = classify(hiddenOutput);
        Neuron outputWinner = competitiveLayer.get(competitiveLayer.indexOf(BMU));

        return outputWinner;
    }
    
    public List<String> getResults(){
        return results;
    }
}
