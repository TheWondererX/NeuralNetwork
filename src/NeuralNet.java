import java.util.ArrayList;
import java.util.List;
import java.util.Random;


public class NeuralNet {

    List<List<Double>> weights = new ArrayList<>();
    List<List<Double>> nodes = new ArrayList<>();
    List<List<Double>> bias_weights = new ArrayList<>();

    Random rand = new Random(System.nanoTime());
    private int iNeurons;
    private int hLayers = 0;
    private double ZERO = 0;
    private double[] input;
    private double[] target;


    public void run(){

        HiddenLayer hl = new HiddenLayer(hLayers, nodes, weights, bias_weights);
        hl.initLayer();

        OutputLayer out = new OutputLayer(nodes, weights, bias_weights);
        out.initLayer();

        Backprop bp = new Backprop(hLayers, nodes, weights, bias_weights, target); // work in progress
        bp.init();
    }


    public void addInputLayer(double[] input) {
        this.input = input;
        this.iNeurons = input.length;

        nodes.add(new ArrayList<Double>()); // adding input layer
        for(int i = 0; i < iNeurons; i++){
            nodes.get(0).add(input[i]);
        }
    }


    public void addHiddenLayer(int hNeurons) {
        this.hLayers += 1;

        nodes.add(new ArrayList<Double>()); // adding hidden layer
        for (int j = 0; j < hNeurons; j++) {
            nodes.get(hLayers).add(ZERO);
        }

    }


    public void addOutputLayer(int oNeurons) {

        nodes.add(new ArrayList<Double>()); // adding output layer
        for(int i = 0; i < oNeurons; i++) {
            nodes.get(nodes.size()-1).add(ZERO);
        }
    }


    public void genWeight() { // weight between layers

        for (int i = 0; i < nodes.size()-1;i++) {
            weights.add(new ArrayList<Double>());
            for (int j = 0;j < nodes.get(i).size() * ((i < nodes.size()-1)? nodes.get(i+1).size(): 1) ; j++) {
                weights.get(i).add(rand.nextDouble() * (1 + 1) - 1); // randomValue = min + (max - min) * rand.nextDouble()
            }
        }
        addBIAS(); // generating bias's weights
    }


    public void showWeight(){

        System.out.println("Weights are: ");
        for (List<Double> list : weights) {

            System.out.println(list);
        }
    }


    public void addBIAS() {

        for(int i = 0; i < nodes.size()-1; i++){
            bias_weights.add(new ArrayList<Double>());
            for(int j = 0; j < nodes.get(i+1).size(); j++){
                    bias_weights.get(i).add(rand.nextDouble() * (1 + 1) - 1);
            }
        }

    }

    public void addTargetValue(double[] target) {
        this.target = target;
    }


    public void newInputValue(int a, int b, int c) // to be improved..
    {
        input[0] = a;
        input[1] = b;
        input[2] = c;


        for(int i = 0; i < iNeurons; i++) {
            nodes.get(0).set(i, input[i]);
        }
    }

    public void newTargetValue(int a, int b, int c) // to be improved..
    {
        target[0] = a;
        target[1] = b;
        target[2] = c;

    }
}


/*
    In Java you would so something like:

        List<List<List<Object>>> listOfListsOfLists =new ArrayList<List<List<Object>>>();

        Then to access the items, you would use:

        listOfListsOfLists.get(a).get(b).get(c);

        Or, to iterate over everything:

        for (List<List<Object>> list2: listOfListsOfLists) {
        for (List<Object> list1: list2) {
        for (Object o: list1) {
        // use `o`
        }
       }
} */