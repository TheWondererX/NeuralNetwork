import java.util.ArrayList;
import java.util.List;


public class Backprop {

    List<List<Double>> nodes = new ArrayList<>();
    List<List<Double>> weights = new ArrayList<>();
    List<List<Double>> bias_weights = new ArrayList<>();
    List<Double> outErr = new ArrayList<>();
    List<List<Double>> hidErr = new ArrayList<>();

    private int wSize;
    private int nSize;
    private int bSize;
    private double ZERO = 0;
    private int hLayers;
    private double p = 0.5; // learning coefficient
    private double[] target;

    public Backprop(int hLayers, List<List<Double>> nodes, List<List<Double>> weights, List<List<Double>> bias_weights, double[] target){

       this.hLayers = hLayers;
       this.nodes = nodes;
       this.weights = weights;
       this.bias_weights = bias_weights;
       this.nSize = nodes.size();
       this.wSize = weights.size();
       this.bSize = bias_weights.size();
       this.target = target;
   }

    public void init(){

        createOurErrArray();
        createHidErrArray();
        outLayerErrorCalc();
        lHiddenLayerErrorCalc();
        outLayerWeightsAdjust();
        otherLayersWeightAdjust();
        //showWeight();

    }

   public void outLayerErrorCalc(){
        double value;

        for(int i = 0; i < nodes.get(nSize-1).size(); i++){
            value = nodes.get(nSize-1).get(i);
            outErr.set(i,((target[i] - value) * (value * (1.0 - value))));
        }
   }

   public void lHiddenLayerErrorCalc(){ // error calculation for last hidden layer
       double value = 0;

       for(int i = 0; i < nodes.get(nSize-2).size(); i++){
           for(int j = 0; j < nodes.get(nSize-1).size(); j++){
               value += outErr.get(j) * weights.get(wSize-1).get(j + i * ((nodes.get(nSize-2).size() <= nodes.get(nSize-1).size()? nodes.get(nSize-2).size(): nodes.get(nSize-1).size())));
           }
           hidErr.get(0).set(i, value * (nodes.get(nSize-2).get(i) * (1 - nodes.get(nSize-2).get(i))));
           value = 0;
       }
   }

   public void hf_lHiddenLayerErrorCalc(int k){ // if more than one hidden layer
        double value = 0;

       for(int i = 0; i < nodes.get(nSize-2-k).size(); i++){
           for(int j = 0; j < nodes.get(nSize-1-k).size(); j++){
                value =+ hidErr.get(k-1).get(j) * weights.get(wSize-1-k).get(j + i * ((nodes.get(nSize-2-k).size() <= nodes.get(nSize-1-k).size()? nodes.get(nSize-2-k).size(): nodes.get(nSize-1-k).size()))); // to be checked
           }
           hidErr.get(k).set(i, value * (nodes.get(nSize-2-k).get(i) * (1 - nodes.get(nSize-2-k).get(i))));
           value = 0;
       }
   }

    public void outLayerWeightsAdjust(){ // the weights between output and last hidden layer
        double value;

        for(int i = 0; i < nodes.get(nSize-1).size(); i++){
            for(int j = 0; j < nodes.get(nSize-2).size(); j++){
                value = weights.get(wSize-1).get(j + (i * (nodes.get(nSize-2).size()))) + (p * outErr.get(i) * nodes.get(nSize-2).get(j));
                weights.get(wSize-1).set((j + (i * (nodes.get(nSize-2).size()))), value);
            }
            value = bias_weights.get(bSize-1).get(i) + (p * outErr.get(i)); // * 1 has been omitted
            bias_weights.get(bSize-1).set(i, value);
        }
    }

    public void otherLayersWeightAdjust(){
        double value;

        for(int i = 0; i < hLayers; i++) {
            for(int j = 0; j < nodes.get(nSize-2-i).size(); j++) {
                for(int k = 0; k < nodes.get(nSize-3-i).size(); k++) {
                    value = weights.get(wSize-2-i).get(k + (j * (nodes.get(nSize-3-i).size()))) + (p * hidErr.get(i).get(j) * nodes.get(nSize-3-i).get(k));
                    weights.get(wSize-2-i).set(k + (j * (nodes.get(nSize-3-i).size())), value);
                }
                value = bias_weights.get(bSize-2-i).get(j) + (p * hidErr.get(i).get(j)); // * 1 has been omitted
                bias_weights.get(bSize-2-i).set(j, value);
            }
            if((i+1) < hLayers) {
                 hf_lHiddenLayerErrorCalc(i+1);
            }
        }
    }


    public void showWeight(){
        System.out.println("\nNew weights are: ");
        for (List<Double> list : weights) {

            System.out.println(list);
        }
    }

    public void createOurErrArray(){
        for(int i = 0; i < nodes.get(nSize-1).size(); i++){
            outErr.add(ZERO);
        }
    }

    public void createHidErrArray(){
        for(int i = 0; i < hLayers; i++){
            hidErr.add(new ArrayList<Double>());
            for(int j = 0; j < nodes.get(nSize-2-i).size(); j++){
                hidErr.get(i).add(ZERO);
            }
        }
    }

}
