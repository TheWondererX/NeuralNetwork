import java.util.ArrayList;
import java.util.List;


public class OutputLayer {

    List<List<Double>> nodes = new ArrayList<>();
    List<List<Double>> weights = new ArrayList<>();
    List<List<Double>> bias_weights = new ArrayList<>();


    OutputLayer(List<List<Double>> nodes, List<List<Double>> weights, List<List<Double>> bias_weights){
        this.nodes = nodes;
        this.weights = weights;
        this.bias_weights = bias_weights;
    }

    public void initLayer(){
        double value = 0;
        int nSize = nodes.size();
        int wSize = weights.size();
        int bSize = bias_weights.size();

        for (int i = 0; i < nodes.get(nSize-1).size(); i++) {
            for (int j = 0; j < nodes.get(nSize-2).size(); j++) {

                value += nodes.get(nSize-2).get(j) * weights.get(wSize-1).get(j + i * (nodes.get(nSize-2).size()));
            }
            value += 1 * bias_weights.get(bSize-1).get(i); // BIAS
            value = activationfunction(value);
            nodes.get(nSize-1).set(i, value);

            value = 0;
        }

        showOutput();

    }

    public double activationfunction(double tmpvalue) {
        return (1/(1+Math.exp(-tmpvalue)));
    }

    public void showOutput(){
        System.out.println("\nLast Nodes values: ");
        for (List<Double> list : nodes) {

            System.out.println(list);
        }
    }
}
