import java.util.ArrayList;
import java.util.List;


public class HiddenLayer {

    List<List<Double>> nodes = new ArrayList<>();
    List<List<Double>> weights = new ArrayList<>();
    List<List<Double>> bias_weights = new ArrayList<>();

    private int hLayers;


    public HiddenLayer(int hLayers, List<List<Double>> nodes, List<List<Double>> weights, List<List<Double>> bias_weights) {
        this.hLayers = hLayers;
        this.nodes = nodes;
        this.weights = weights;
        this.bias_weights = bias_weights;
    }

    public void initLayer() {
        double value = 0;

        for (int i = 0; i < hLayers; i++) {
            for (int j = 0; j < nodes.get(i+1).size(); j++) {
                for (int k = 0; k < nodes.get(i).size(); k++) {

                    value += nodes.get(i).get(k) * weights.get(i).get(k + j * (nodes.get(i).size()));
                }
                value += 1 * bias_weights.get(i).get(j); // BIAS
                value = activationfunction(value);
                nodes.get(i+1).set(j, value);
                value = 0;
            }
        }

    }

    public double activationfunction(double tmpvalue) {
        return (1/(1+Math.exp(-tmpvalue)));
    }

}