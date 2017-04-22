

public class Main {

    static int Epoch = 10000;
    public static void main(String[] args){

        double[] input = {0, 1, 0}; // input values sample
        double[] target = {1, 0, 1}; // same size as output layer (sample values)


        NeuralNet net = new NeuralNet();
        net.addInputLayer(input);
        //net.addHiddenLayer(2); // adding one hidden layer with 2 neurons
        net.addHiddenLayer(3); // adding one hidden layer with 3 neurons
        net.addOutputLayer(3); // adding output layer with 4 neurons
        net.addTargetValue(target);
        net.genWeight(); // generating weight's (and BIAS)
        //net.showWeight();

        //for test purpose only (binary)
        for(int i = 0; i < Epoch; i++){
            net.newInputValue(0, 0, 0);  // for 1
            net.newTargetValue(0, 0, 1);
            net.run();

            net.newInputValue(0, 0, 1);  // for 2
            net.newTargetValue(0, 1, 0);
            net.run();

            net.newInputValue(0, 1, 0);  // for 3 and so on..
            net.newTargetValue(0, 1, 1);
            net.run();

            net.newInputValue(0, 1, 1);
            net.newTargetValue(1, 0, 0);
            net.run();

            net.newInputValue(1, 0, 0);
            net.newTargetValue(1, 0, 1);
            net.run();

            net.newInputValue(1, 0, 1);
            net.newTargetValue(1, 1, 0);
            net.run();

            net.newInputValue(1, 1, 0); // for 7
            net.newTargetValue(1, 1, 1);
            net.run();

            net.newInputValue(1, 1, 1); // reset to 0
            net.newTargetValue(0, 0, 0);
            net.run();

            System.out.println("\nEpoch №" + (i+1) + " ended.");
        }
    }

}
