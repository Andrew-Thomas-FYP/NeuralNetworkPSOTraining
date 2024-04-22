using System;

public class NeuralNetwork
{
    private int[] layers; //array of layers in the network
    private double[][] neurons; // array of neurons in each layer of the network
    private double[][] biases; // array of biases in each layer of the network
    private double[][][] weights; //array of weights for each neuron of each layer of the network
    private int totalWeightsAndBiases;


    //constructor
    public NeuralNetwork(int[] layers)
	{
        this.layers = new int[layers.Length];
        for (int i = 0; i < layers.Length; i++)
        {
            this.layers[i] = layers[i];
        }
        totalWeightsAndBiases = GetTotalWeightsAndBiases();
        InitNeurons();
        InitBiases();
        InitWeights();
	}

    //initialises the weights to random numbers for the initial pass through of the network
    private void InitWeights()
    {
        Random rnd = new Random();
        List<double[][]> weightsList = new List<double[][]>();
        for (int i = 1; i < layers.Length; i++)
        {
            List<double[]> layerWeightsList = new List<double[]>();
            int previousNeurons = layers[i - 1];
            for (int j = 0; j < neurons[i].Length; j++)
            {
                double[] neuronWeights = new double[previousNeurons];
                // put the for loop back in
                layerWeightsList.Add(neuronWeights);
            }
            weightsList.Add(layerWeightsList.ToArray());
        }
        weights = weightsList.ToArray();
    }

    //initialises the biases for each layer of the network to random numbers between -0.5 and 0.5
    private void InitBiases()
    {
        Random rnd = new Random();
        List<double[]> biasList = new List<double[]>();
        for (int i = 0; i < layers.Length; i++)
        {
            double[] bias = new double[layers[i]];

            biasList.Add(bias);
        }
        biases = biasList.ToArray();
    }

    //initialises the total neurons for the network based on number of layers
    private void InitNeurons()
    {
        List<double[]> neuronsList = new List<double[]>();
        for (int i = 0; i < layers.Length; i++)
        {
            neuronsList.Add(new double[layers[i]]);
        }
        neurons = neuronsList.ToArray();
    }

    public double Activation(double val)
    {
        return (double)Math.Tanh(val);
    }


    public double[] FeedForward(double[] inputs)
    {
        // Ensure the number of inputs matches the number of neurons in the input layer
        if (inputs.Length != neurons[0].Length)
        {
            throw new ArgumentException("The length of inputs does not match the number of neurons in the input layer.");
        }

        // Input layer
        for (int i = 0; i < inputs.Length; i++)
        {
            neurons[0][i] = inputs[i];
        }

        // Hidden layers and output layer
        for (int i = 1; i < layers.Length; i++)
        {
            for (int j = 0; j < neurons[i].Length; j++)
            {
                double val = 0f;

                // Sum of product of weights and neurons from previous layer
                for (int k = 0; k < neurons[i - 1].Length; k++)
                {
                    val += weights[i - 1][j][k] * neurons[i - 1][k];
                }

                // Activation function
                neurons[i][j] = Activation(val + biases[i][j]);
            }
        }
        return neurons[neurons.Length - 1];
    }
    
    public double[] Train(double[][] trainingData, int numParticles, int maxEpochs, double exitError)
    {
        Random rnd = new Random();

        
        int epoch = 0;
        double minX = -10;
        double maxX = 10;
        double w = 0.4;
        double c1 = 1.49445;
        double c2 = 1.49445;
        double r1, r2;

        Particle[] swarm = new Particle[numParticles];

        double[] gbest = new double[totalWeightsAndBiases]; //best global position
        double gebest = double.MaxValue; //best global error

        for (int i = 0; i < swarm.Length; i++)
        {
            double[] startingPos = new double[totalWeightsAndBiases];
            for (int j = 0; j < startingPos.Length; j++)
            {
                startingPos[j] += (rnd.NextDouble() * 2 - 1) * 0.1;
            }

            double error = MeanSqrError(trainingData, startingPos);

            double[] startingVelocity = new double[totalWeightsAndBiases];

            for (int j = 0; j < startingVelocity.Length; j++)
            {
                double lo = 0.1 * minX;
                double hi = 0.1 * maxX;
                startingVelocity[j] = (hi - lo) * rnd.NextDouble() + lo;
            }

            swarm[i] = new Particle(startingPos, startingVelocity, startingPos, error, error);

            if (swarm[i].error < gebest)
            {
                gebest = swarm[i].error;
                swarm[i].position.CopyTo(gbest, 0);
            }
        }
        while (epoch < maxEpochs && gebest > exitError)
        {
            foreach (var particle in swarm)
            {
                double newError = MeanSqrError(trainingData, particle.position);

                // if the new position is better, update the personal best
                if (newError < particle.ebest)
                {
                    particle.ebest = newError;
                    particle.position.CopyTo(particle.pbest, 0);
                }

                // if the new position is the new global best, update the global best
                if (newError < gebest)
                {
                    gebest = newError;
                    particle.position.CopyTo(gbest, 0);
                }
            }

            // update velocity and position of each particle
            foreach (var particle in swarm)
            {
                for (int i = 0; i < particle.velocity.Length; i++)
                {
                    r1 = rnd.NextDouble();
                    r2 = rnd.NextDouble();
                    particle.velocity[i] = (w * particle.velocity[i]) +
                                           (c1 * r1 * (particle.pbest[i] - particle.position[i])) +
                                           (c2 * r2 * (gbest[i] - particle.position[i]));

                    particle.position[i] += particle.velocity[i];
                }
            }
            epoch++;
        }

        // copy the weights and biases from the best position found
        Array.Copy(gbest, 0, weights, 0, weights.Length);
        Array.Copy(gbest, weights.Length, biases, 0, biases.Length);

        return gbest;
    }

    private double MeanSqrError(double[][] trainingData, double[] weightsAndBiases)
    {
        // Check if the length of weightsAndBiases is correct
        if (weightsAndBiases.Length != totalWeightsAndBiases)
        {
            throw new ArgumentException("The length of weightsAndBiases is incorrect.");
        }

        // Save original weights and biases
        var originalWeights = weights;
        var originalBiases = biases;

        try
        {
            // Convert the weightsAndBiases back to weights and biases
            int weightCount = 0;
            for (int i = 0; i < weights.Length; ++i)
            {
                for (int j = 0; j < weights[i].Length; ++j)
                {
                    for (int k = 0; k < weights[i][j].Length; ++k)
                    {
                        if (weightCount < weightsAndBiases.Length)
                        {
                            weights[i][j][k] = weightsAndBiases[weightCount++];
                        }
                    }
                }
            }

            int biasCount = weightCount;
            for (int i = 0; i < biases.Length; ++i)
            {
                for (int j = 0; j < biases[i].Length; ++j)
                {
                    if(biasCount < weightsAndBiases.Length)
                    {
                        biases[i][j] = weightsAndBiases[biasCount++];
                    }
                    
                }
            }

            // Compute error with the given weights and biases
            double sumError = 0;
            for (int i = 0; i < trainingData.Length; ++i)
            {

                double[] inputs = trainingData[i].Skip(1).ToArray();
                Array.Copy(trainingData[i], 1, inputs, 0, inputs.Length);  // Exclude the label (first element)
                int trueValue = (int)trainingData[i][0];  // True label is the first element

                // Exclude the first element, which is the label


                double expected = trainingData[i][0];

                double[] outputs = FeedForward(inputs);
                int predictedValue = Array.IndexOf(outputs, outputs.Max());  // The index of the highest value is taken as the predicted label


                for (int j = 0; j < outputs.Length; j++)
                {
                    double error = (j == trueValue ? 1.0 : 0.0) - outputs[j];
                    sumError += error * error;
                }
            }

            return sumError / trainingData.Length;
        }
        finally
        {
            // Restore original weights and biases
            weights = originalWeights;
            biases = originalBiases;
        }
    }

    private int GetTotalWeightsAndBiases()
    {
        int totalWeights = 0;
        int totalBiases = 0;

        for (int i = 0; i < layers.Length - 1; ++i)
        {
            totalWeights += layers[i] * layers[i + 1]; // the weights between two layers are layer[i] * layer[i + 1]
            totalBiases += layers[i + 1]; // the biases in a layer are equal to the number of neurons in that layer
        }

        return totalWeights + totalBiases;
    }


    public void SetWeights(double[] weights)
    {
        int index = 0;

        for (int i = 0; i < layers.Length - 1; i++)
        {
            for (int j = 0; j < neurons[i].Length; j++)
            {
                for (int k = 0; k < neurons[i + 1].Length; k++)
                {
                    this.weights[i][j][k] = weights[index++];
                }
            }
        }
    }
}
