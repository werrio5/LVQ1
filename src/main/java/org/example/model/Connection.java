package org.example.model;

public class Connection {

    private Neuron startNode;
    private Neuron endNode;
    private double weight;

    public Connection(Neuron startNode, Neuron endNode, double weight){
        this.startNode = startNode;
        this.endNode = endNode;
        this.weight = weight;
    }

    public Neuron getStartNode() {
        return startNode;
    }

    public void setStartNode(Neuron startNode) {
        this.startNode = startNode;
    }

    public Neuron getEndNode() {
        return endNode;
    }

    public void setEndNode(Neuron endNode) {
        this.endNode = endNode;
    }

    public double getWeight() {
        return weight;
    }

    public void setWeight(double weight) {
        this.weight = weight;
    }
}
