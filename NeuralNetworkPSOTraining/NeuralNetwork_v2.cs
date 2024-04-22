using Microsoft.Win32;
using System.Diagnostics;
using Bytescout.Spreadsheet;
using System.IO;
using OfficeOpenXml;

public class NeuralNetwork_v2
{
    private int numInput; // Number of nodes in the input layer
    private int numHidden; // Number of nodes in the hidden layer
    private int numOutput; // Number of nodes in the output layer

    private double[] inputs; // Inputs to the network

    private double[][] ihWeights; // Weights for the connections between input and hidden layers
    private double[] hBiases; // Biases for the hidden layer nodes
    private double[] hOutputs; // Outputs after the activation function is applied in the hidden layer nodes

    private double[][] hoWeights; // Weights for the connections between hidden and output layers
    private double[] oBiases; // Biases for the output layer nodes
    private double[] outputs; // Final outputs of the network

    public NeuralNetwork_v2(int numInput, int numHidden, int numOutput)
    {
        this.numInput = numInput;
        this.numHidden = numHidden;
        this.numOutput = numOutput;

        // Initialize inputs
        inputs = new double[this.numInput];

        // Initialize weights and biases between input and hidden layers
        ihWeights = MakeMatrix(this.numInput, this.numHidden);
        hBiases = new double[this.numHidden];

        // Initialize output values for hidden layer
        hOutputs = new double[this.numHidden];

        // Initialize weights and biases between hidden and output layers
        hoWeights = MakeMatrix(this.numHidden, this.numOutput);
        oBiases = new double[this.numOutput];

        // Initialize final outputs
        outputs = new double[this.numOutput];
    }

    private double[][] MakeMatrix(int rows, int cols)
    {
        var result = new double[rows][];
        for (int r = 0; r < result.Length; ++r)
            result[r] = new double[cols];
        return result;
    }

    public double[] Feedforward(double[] inputArray)
    {
        // Ensure the input count matches the number of input nodes
        if (inputArray.Length != numInput)
        {
            throw new Exception("Number of inputs must be the same as the number of input nodes.");
        }

        // Assign input to the input array
        this.inputs = inputArray;

        // Propagate values from input layer to hidden layer
        for (int j = 0; j < numHidden; j++)
        {
            double sum = 0.0;
            for (int i = 0; i < numInput; i++)
            {
                sum += this.inputs[i] * this.ihWeights[i][j];
            }

            sum += this.hBiases[j];
            this.hOutputs[j] = Math.Max(0, sum); // ReLU activation
        }

        // Propagate values from hidden layer to output layer
        for (int j = 0; j < numOutput; j++)
        {
            double sum = 0.0;
            for (int i = 0; i < numHidden; i++)
            {
                sum += this.hOutputs[i] * this.hoWeights[i][j];
            }

            sum += this.oBiases[j];
            this.outputs[j] = sum; // Apply softmax later after calculating all sums
        }

        // Apply softmax activation
        this.outputs = Softmax(this.outputs);

        return this.outputs;
    }

    private double[] Softmax(double[] outputs)
    {
        double max = outputs.Max();

        double scale = 0.0;
        for (int i = 0; i < outputs.Length; i++)
        {
            scale += Math.Exp(outputs[i] - max);
        }

        double[] result = new double[outputs.Length];
        for (int i = 0; i < outputs.Length; i++)
        {
            result[i] = Math.Exp(outputs[i] - max) / scale;
        }

        return result;
    }


    public double[] TrainStar(double[][] trainingData, int numParticles, int maxIterations, double exitError, TimeSpan exitTime, double[][] testData, TimeSpan interval)
    {
        int totalWeights = (this.numInput * this.numHidden) + (this.numHidden * this.numOutput) +
        this.numHidden + this.numOutput;

        System.Diagnostics.Stopwatch stopwatch = System.Diagnostics.Stopwatch.StartNew();
        stopwatch.Stop();
        TimeSpan timeElapsed = stopwatch.Elapsed;
        
        TimeSpan counter = new TimeSpan(00, 00, 00);

        Random rnd = new Random();

        int iteration = 0;
        double minX = -10;
        double maxX = 10;
        double w = 0.4;
        double c1 = 1.49445;
        double c2 = 1.49445;
        double r1, r2;
        String top = "Star";
        var accuracyList = new List<double>();
        var timeList = new List<string>();
        int accuracyCounter = 0;

        Particle[] swarm = new Particle[numParticles];

        double[] gbest = new double[totalWeights]; //best global position
        double gebest = double.MaxValue; //best global error

        for (int i = 0; i < swarm.Length; i++)
        {
            double[] startingPos = new double[totalWeights];
            for (int j = 0; j < startingPos.Length; j++)
            {
                startingPos[j] += (rnd.NextDouble() * 2 - 1) * 0.1;
            }

            double error = MeanSqrError(trainingData, startingPos);

            double[] startingVelocity = new double[totalWeights];

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
        while (timeElapsed < exitTime)
        {
            stopwatch.Start();
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
            iteration++;
            stopwatch.Stop();
            timeElapsed = stopwatch.Elapsed;

            if(timeElapsed > counter)
            {
                this.SetWeights(gbest);
                double accuracy = TestAccuracy(testData);
                

                string elapsedTime = String.Format("{0:00}:{1:00}:{2:00}.{3:00}",
                timeElapsed.Hours, timeElapsed.Minutes, timeElapsed.Seconds,
                timeElapsed.Milliseconds / 10);



                accuracyList.Add(accuracy);
                timeList.Add(elapsedTime);

                if (accuracyCounter > 0) {
                    if (accuracyList[accuracyCounter] < accuracyList[accuracyCounter - 1])
                    {
                        accuracyList[accuracyCounter] = accuracyList[accuracyCounter - 1];
                    }
                }

                accuracyCounter++;

                counter = counter.Add(interval);
            }
            
        }

        this.SetWeights(gbest);  
        double[] result = new double[totalWeights];
        Array.Copy(gbest, result, result.Length);
        SaveToExcel(timeList, accuracyList, top);
        return result;
    }

    public double[] TrainCircle(double[][] trainingData, int numParticles, int maxIterations, double exitError, TimeSpan exitTime, double[][] testData, TimeSpan interval)
    {
        int totalWeights = (this.numInput * this.numHidden) + (this.numHidden * this.numOutput) +
        this.numHidden + this.numOutput;

        Random rnd = new Random();

        int iteration = 0;
        double minX = -10;
        double maxX = 10;
        double w = 0.4;
        double c1 = 1.49445;
        double c2 = 1.49445;
        double r1, r2;
        String top = "Circle";
        var accuracyList = new List<double>();
        var timeList = new List<string>();
        int accuracyCounter = 0;

        System.Diagnostics.Stopwatch stopwatch = System.Diagnostics.Stopwatch.StartNew();
        stopwatch.Stop();
        TimeSpan timeElapsed = stopwatch.Elapsed;

        TimeSpan counter = new TimeSpan(00, 00, 00);

        Particle[] swarm = new Particle[numParticles];

        double[] gbest = new double[totalWeights]; //best global position
        double gebest = double.MaxValue; //best global error

        for (int i = 0; i < swarm.Length; i++)
        {
            double[] startingPos = new double[totalWeights];
            for (int j = 0; j < startingPos.Length; j++)
            {
                startingPos[j] += (rnd.NextDouble() * 2 - 1) * 0.1;
            }

            double error = MeanSqrError(trainingData, startingPos);

            double[] startingVelocity = new double[totalWeights];

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
        while (timeElapsed < exitTime)
        {
            stopwatch.Start();
            for (int i = 0; i < swarm.Length; i++)
            {
                var particle = swarm[i];
                var leftNeighbour = swarm[(i - 1 + numParticles) % numParticles];
                var rightNeighbour = swarm[(i + 1) % numParticles];

                double newError = MeanSqrError(trainingData, particle.position);

                // if the new position is better, update the personal best
                if (newError < particle.ebest)
                {
                    particle.ebest = newError;
                    particle.position.CopyTo(particle.pbest, 0);
                }

                // Update local best position considering neighbours
                double[] lbest;  // local best position
                double elbest;   // local best error

                if (leftNeighbour.ebest < rightNeighbour.ebest)
                {
                    lbest = leftNeighbour.pbest;
                    elbest = leftNeighbour.ebest;
                }
                else
                {
                    lbest = rightNeighbour.pbest;
                    elbest = rightNeighbour.ebest;
                }

                // if the new position is the new global best, update the global best
                if (newError < gebest)
                {
                    gebest = newError;
                    particle.position.CopyTo(gbest, 0);
                }

                // update velocity and position of the particle
                for (int j = 0; j < particle.velocity.Length; j++)
                {
                    r1 = rnd.NextDouble();
                    r2 = rnd.NextDouble();

                    // Use local best here instead of global best
                    particle.velocity[j] = (w * particle.velocity[j]) +
                                           (c1 * r1 * (particle.pbest[j] - particle.position[j])) +
                                           (c2 * r2 * (lbest[j] - particle.position[j]));

                    particle.position[j] += particle.velocity[j];
                }
            }
            iteration++;

            stopwatch.Stop();
            timeElapsed = stopwatch.Elapsed;

            if (timeElapsed > counter)
            {
                this.SetWeights(gbest);
                double accuracy = TestAccuracy(testData);
                

                string elapsedTime = String.Format("{0:00}:{1:00}:{2:00}.{3:00}",
                timeElapsed.Hours, timeElapsed.Minutes, timeElapsed.Seconds,
                timeElapsed.Milliseconds / 10);



                accuracyList.Add(accuracy);
                timeList.Add(elapsedTime);

                if (accuracyCounter > 0)
                {
                    if (accuracyList[accuracyCounter] < accuracyList[accuracyCounter - 1])
                    {
                        accuracyList[accuracyCounter] = accuracyList[accuracyCounter - 1];
                    }
                }

                accuracyCounter++;

                counter = counter.Add(interval);
            }
        }

        this.SetWeights(gbest);
        double[] result = new double[totalWeights];
        Array.Copy(gbest, result, result.Length);
        SaveToExcel(timeList, accuracyList, top);
        return result;
    }

    public double[] TrainVonNeumann(double[][] trainingData, int numParticles, int maxIterations, double exitError, int gridDimension, TimeSpan exitTime, double[][] testData, TimeSpan interval)
    {
        int totalWeights = (this.numInput * this.numHidden) + (this.numHidden * this.numOutput) +
        this.numHidden + this.numOutput;

        System.Diagnostics.Stopwatch stopwatch = System.Diagnostics.Stopwatch.StartNew();
        stopwatch.Stop();
        TimeSpan timeElapsed = stopwatch.Elapsed;

        TimeSpan counter = new TimeSpan(00, 00, 00);

        Random rnd = new Random();

        int iteration = 0;
        double minX = -10;
        double maxX = 10;
        double w = 0.4;
        double c1 = 1.49445;
        double c2 = 1.49445;
        double r1, r2;
        String top = "VN";
        var accuracyList = new List<double>();
        var timeList = new List<string>();
        int accuracyCounter = 0;

        // Create grid for von Neumann topology
        Particle[,] grid = new Particle[gridDimension, gridDimension];
        Particle[] swarm = new Particle[numParticles];

        double[] gbest = new double[totalWeights]; //best global position
        double gebest = double.MaxValue; //best global error

        // Initialize particles and assign to grid
        for (int p = 0; p < swarm.Length; p++)
        {
            double[] startingPos = new double[totalWeights];
            for (int j = 0; j < startingPos.Length; j++)
            {
                startingPos[j] += (rnd.NextDouble() * 2 - 1) * 0.1;
            }

            double error = MeanSqrError(trainingData, startingPos);

            double[] startingVelocity = new double[totalWeights];

            for (int j = 0; j < startingVelocity.Length; j++)
            {
                double lo = 0.1 * minX;
                double hi = 0.1 * maxX;
                startingVelocity[j] = (hi - lo) * rnd.NextDouble() + lo;
            }

            swarm[p] = new Particle(startingPos, startingVelocity, startingPos, error, error);
            grid[p / gridDimension, p % gridDimension] = swarm[p];

            if (swarm[p].error < gebest)
            {
                gebest = swarm[p].error;
                swarm[p].position.CopyTo(gbest, 0);
            }
        }
        while (timeElapsed < exitTime)
        {
            stopwatch.Start();

            for (int i = 0; i < gridDimension; i++)
            {
                for (int j = 0; j < gridDimension; j++)
                {
                    Particle particle = grid[i, j];
                    double newError = MeanSqrError(trainingData, particle.position);

                    if (newError < particle.ebest)
                    {
                        particle.ebest = newError;
                        particle.position.CopyTo(particle.pbest, 0);
                    }

                    // Compute local best for Von Neumann neighbourhood
                    Particle[] neighbours = new Particle[4];
                    neighbours[0] = grid[(i - 1 + gridDimension) % gridDimension, j]; // North
                    neighbours[1] = grid[i, (j + 1) % gridDimension]; // East
                    neighbours[2] = grid[(i + 1) % gridDimension, j]; // South
                    neighbours[3] = grid[i, (j - 1 + gridDimension) % gridDimension]; // West

                    Particle lbestParticle = neighbours.OrderBy(x => x.ebest).First();
                    double[] lbest = lbestParticle.pbest;
                    double elbest = lbestParticle.ebest;

                    if (newError < gebest)
                    {
                        gebest = newError;
                        particle.position.CopyTo(gbest, 0);
                    }

                    for (int k = 0; k < particle.velocity.Length; k++)
                    {
                        r1 = rnd.NextDouble();
                        r2 = rnd.NextDouble();

                        particle.velocity[k] = (w * particle.velocity[k]) +
                                               (c1 * r1 * (particle.pbest[k] - particle.position[k])) +
                                               (c2 * r2 * (lbest[k] - particle.position[k]));

                        particle.position[k] += particle.velocity[k];
                    }
                }
            }
            iteration++;
            stopwatch.Stop();
            timeElapsed = stopwatch.Elapsed;

            if (timeElapsed > counter)
            {
                this.SetWeights(gbest);
                double accuracy = TestAccuracy(testData);
                

                string elapsedTime = String.Format("{0:00}:{1:00}:{2:00}.{3:00}",
                timeElapsed.Hours, timeElapsed.Minutes, timeElapsed.Seconds,
                timeElapsed.Milliseconds / 10);



                accuracyList.Add(accuracy);
                timeList.Add(elapsedTime);

                if (accuracyCounter > 0)
                {
                    if (accuracyList[accuracyCounter] < accuracyList[accuracyCounter - 1])
                    {
                        accuracyList[accuracyCounter] = accuracyList[accuracyCounter - 1];
                    }
                }

                accuracyCounter++;

                counter = counter.Add(interval);
            }
        }

        this.SetWeights(gbest);
        double[] result = new double[totalWeights];
        Array.Copy(gbest, result, result.Length);
        SaveToExcel(timeList, accuracyList, top);
        return result;
    }

    public double[] TrainRandomCircle(double[][] trainingData, int numParticles, int maxIterations, double exitError, TimeSpan exitTime, double[][] testData, TimeSpan interval)
    {
        int totalWeights = (this.numInput * this.numHidden) + (this.numHidden * this.numOutput) +
        this.numHidden + this.numOutput;

        System.Diagnostics.Stopwatch stopwatch = System.Diagnostics.Stopwatch.StartNew();
        stopwatch.Stop();
        TimeSpan timeElapsed = stopwatch.Elapsed;

        TimeSpan counter = new TimeSpan(00, 00, 00);

        Random rnd = new Random();

        int iteration = 0;
        double minX = -10;
        double maxX = 10;
        double w = 0.4;
        double c1 = 1.49445;
        double c2 = 1.49445;
        double r1, r2;
        String top = "RCircle";
        var accuracyList = new List<double>();
        var timeList = new List<string>();
        int accuracyCounter = 0;

        Particle[] swarm = new Particle[numParticles];

        double[] gbest = new double[totalWeights]; //best global position
        double gebest = double.MaxValue; //best global error

        for (int i = 0; i < swarm.Length; i++)
        {
            double[] startingPos = new double[totalWeights];
            for (int j = 0; j < startingPos.Length; j++)
            {
                startingPos[j] += (rnd.NextDouble() * 2 - 1) * 0.1;
            }

            double error = MeanSqrError(trainingData, startingPos);

            double[] startingVelocity = new double[totalWeights];

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
        while (timeElapsed < exitTime)
        {
            stopwatch.Start();
            swarm = Shuffle(swarm, rnd);

            for (int i = 0; i < swarm.Length; i++)
            {
                var particle = swarm[i];
                var leftNeighbour = swarm[(i - 1 + numParticles) % numParticles];
                var rightNeighbour = swarm[(i + 1) % numParticles];

                double newError = MeanSqrError(trainingData, particle.position);

                // if the new position is better, update the personal best
                if (newError < particle.ebest)
                {
                    particle.ebest = newError;
                    particle.position.CopyTo(particle.pbest, 0);
                }

                // Update local best position considering neighbours
                double[] lbest;  // local best position
                double elbest;   // local best error

                if (leftNeighbour.ebest < rightNeighbour.ebest)
                {
                    lbest = leftNeighbour.pbest;
                    elbest = leftNeighbour.ebest;
                }
                else
                {
                    lbest = rightNeighbour.pbest;
                    elbest = rightNeighbour.ebest;
                }

                // if the new position is the new global best, update the global best
                if (newError < gebest)
                {
                    gebest = newError;
                    particle.position.CopyTo(gbest, 0);
                }

                // update velocity and position of the particle
                for (int j = 0; j < particle.velocity.Length; j++)
                {
                    r1 = rnd.NextDouble();
                    r2 = rnd.NextDouble();

                    // Use local best here instead of global best
                    particle.velocity[j] = (w * particle.velocity[j]) +
                                           (c1 * r1 * (particle.pbest[j] - particle.position[j])) +
                                           (c2 * r2 * (lbest[j] - particle.position[j]));

                    particle.position[j] += particle.velocity[j];
                }
            }
            iteration++;
            stopwatch.Stop();
            timeElapsed = stopwatch.Elapsed;

            if (timeElapsed > counter)
            {
                this.SetWeights(gbest);
                double accuracy = TestAccuracy(testData);
                

                string elapsedTime = String.Format("{0:00}:{1:00}:{2:00}.{3:00}",
                timeElapsed.Hours, timeElapsed.Minutes, timeElapsed.Seconds,
                timeElapsed.Milliseconds / 10);



                accuracyList.Add(accuracy);
                timeList.Add(elapsedTime);

                if (accuracyCounter > 0)
                {
                    if (accuracyList[accuracyCounter] < accuracyList[accuracyCounter - 1])
                    {
                        accuracyList[accuracyCounter] = accuracyList[accuracyCounter - 1];
                    }
                }

                accuracyCounter++;

                counter = counter.Add(interval);
            }
        }

        this.SetWeights(gbest);
        double[] result = new double[totalWeights];
        Array.Copy(gbest, result, result.Length);
        SaveToExcel(timeList, accuracyList, top);
        return result;
    }

    private Particle[] Shuffle(Particle[] swarm, Random rnd)
    {
        for (int i = swarm.Length - 1; i > 0; i--)
        {
            int index = rnd.Next(i + 1);
            var temp = swarm[index];
            swarm[index] = swarm[i];
            swarm[i] = temp;
        }

        return swarm;
    }

    public double[] TrainEuclidianCircle(double[][] trainingData, int numParticles, int maxIterations, double exitError, int mod, TimeSpan exitTime, double[][] testData, TimeSpan interval)
    {
        int totalWeights = (this.numInput * this.numHidden) + (this.numHidden * this.numOutput) +
        this.numHidden + this.numOutput;

        System.Diagnostics.Stopwatch stopwatch = System.Diagnostics.Stopwatch.StartNew();
        stopwatch.Stop();
        TimeSpan timeElapsed = stopwatch.Elapsed;

        TimeSpan counter = new TimeSpan(00, 00, 00);

        Random rnd = new Random();

        int iteration = 0;
        double minX = -10;
        double maxX = 10;
        double w = 0.4;
        double c1 = 1.49445;
        double c2 = 1.49445;
        double r1, r2;
        String top = "ECircle";
        var accuracyList = new List<double>();
        var timeList = new List<string>();
        int accuracyCounter = 0;

        Particle[] swarm = new Particle[numParticles];

        double[] gbest = new double[totalWeights]; //best global position
        double gebest = double.MaxValue; //best global error

        for (int i = 0; i < swarm.Length; i++)
        {
            double[] startingPos = new double[totalWeights];
            for (int j = 0; j < startingPos.Length; j++)
            {
                startingPos[j] += (rnd.NextDouble() * 2 - 1) * 0.1;
            }

            double error = MeanSqrError(trainingData, startingPos);

            double[] startingVelocity = new double[totalWeights];

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
        while (timeElapsed < exitTime)
        {
            stopwatch.Start();

            if (iteration % mod == 0)
            {
                OrderSwarmBasedOnPosition(swarm);
            }

            for (int i = 0; i < swarm.Length; i++)
            {
                var particle = swarm[i];
                var leftNeighbour = swarm[(i - 1 + numParticles) % numParticles];
                var rightNeighbour = swarm[(i + 1) % numParticles];

                double newError = MeanSqrError(trainingData, particle.position);

                // if the new position is better, update the personal best
                if (newError < particle.ebest)
                {
                    particle.ebest = newError;
                    particle.position.CopyTo(particle.pbest, 0);
                }

                // Update local best position considering neighbours
                double[] lbest;  // local best position
                double elbest;   // local best error

                if (leftNeighbour.ebest < rightNeighbour.ebest)
                {
                    lbest = leftNeighbour.pbest;
                    elbest = leftNeighbour.ebest;
                }
                else
                {
                    lbest = rightNeighbour.pbest;
                    elbest = rightNeighbour.ebest;
                }

                // if the new position is the new global best, update the global best
                if (newError < gebest)
                {
                    gebest = newError;
                    particle.position.CopyTo(gbest, 0);
                }

                // update velocity and position of the particle
                for (int j = 0; j < particle.velocity.Length; j++)
                {
                    r1 = rnd.NextDouble();
                    r2 = rnd.NextDouble();

                    // Use local best here instead of global best
                    particle.velocity[j] = (w * particle.velocity[j]) +
                                           (c1 * r1 * (particle.pbest[j] - particle.position[j])) +
                                           (c2 * r2 * (lbest[j] - particle.position[j]));

                    particle.position[j] += particle.velocity[j];
                }
            }
            iteration++;
            stopwatch.Stop();
            timeElapsed = stopwatch.Elapsed;

            if (timeElapsed > counter)
            {
                this.SetWeights(gbest);
                double accuracy = TestAccuracy(testData);
                

                string elapsedTime = String.Format("{0:00}:{1:00}:{2:00}.{3:00}",
                timeElapsed.Hours, timeElapsed.Minutes, timeElapsed.Seconds,
                timeElapsed.Milliseconds / 10);



                accuracyList.Add(accuracy);
                timeList.Add(elapsedTime);

                if (accuracyCounter > 0)
                {
                    if (accuracyList[accuracyCounter] < accuracyList[accuracyCounter - 1])
                    {
                        accuracyList[accuracyCounter] = accuracyList[accuracyCounter - 1];
                    }
                }

                accuracyCounter++;

                counter = counter.Add(interval);
            }
        }

        this.SetWeights(gbest);
        double[] result = new double[totalWeights];
        Array.Copy(gbest, result, result.Length);
        SaveToExcel(timeList, accuracyList, top);
        return result;
    }

    public double[] TrainEuclideanVonNeumann(double[][] trainingData, int numParticles, int maxIterations, double exitError, int gridDimension, int mod, TimeSpan exitTime, double[][] testData, TimeSpan interval)
    {
        int totalWeights = (this.numInput * this.numHidden) + (this.numHidden * this.numOutput) +
        this.numHidden + this.numOutput;

        System.Diagnostics.Stopwatch stopwatch = System.Diagnostics.Stopwatch.StartNew();
        stopwatch.Stop();
        TimeSpan timeElapsed = stopwatch.Elapsed;

        TimeSpan counter = new TimeSpan(00, 00, 00);

        Random rnd = new Random();

        int iteration = 0;
        double minX = -10;
        double maxX = 10;
        double w = 0.4;
        double c1 = 1.49445;
        double c2 = 1.49445;
        double r1, r2;
        String top = "EVN";
        var accuracyList = new List<double>();
        var timeList = new List<string>();
        int accuracyCounter = 0;

        // Create grid for von Neumann topology
        Particle[,] grid = new Particle[gridDimension, gridDimension];
        Particle[] swarm = new Particle[numParticles];

        double[] gbest = new double[totalWeights]; //best global position
        double gebest = double.MaxValue; //best global error

        // Initialize particles and assign to grid
        for (int p = 0; p < swarm.Length; p++)
        {
            double[] startingPos = new double[totalWeights];
            for (int j = 0; j < startingPos.Length; j++)
            {
                startingPos[j] += (rnd.NextDouble() * 2 - 1) * 0.1;
            }

            double error = MeanSqrError(trainingData, startingPos);

            double[] startingVelocity = new double[totalWeights];

            for (int j = 0; j < startingVelocity.Length; j++)
            {
                double lo = 0.1 * minX;
                double hi = 0.1 * maxX;
                startingVelocity[j] = (hi - lo) * rnd.NextDouble() + lo;
            }


            swarm[p] = new Particle(startingPos, startingVelocity, startingPos, error, error);
            grid[p / gridDimension, p % gridDimension] = swarm[p];

            if (swarm[p].error < gebest)
            {
                gebest = swarm[p].error;
                swarm[p].position.CopyTo(gbest, 0);
            }
        }

        while (timeElapsed < exitTime)
        {
            stopwatch.Start();

            for (int i = 0; i < gridDimension; i++)
            {
                for (int j = 0; j < gridDimension; j++)
                {
                    Particle particle = grid[i, j];
                    double newError = MeanSqrError(trainingData, particle.position);

                    if (newError < particle.ebest)
                    {
                        particle.ebest = newError;
                        particle.position.CopyTo(particle.pbest, 0);
                    }

                    // Compute local best for Von Neumann neighbourhood
                    Particle[] neighbours = new Particle[4];
                    neighbours[0] = grid[(i - 1 + gridDimension) % gridDimension, j]; // North
                    neighbours[1] = grid[i, (j + 1) % gridDimension]; // East
                    neighbours[2] = grid[(i + 1) % gridDimension, j]; // South
                    neighbours[3] = grid[i, (j - 1 + gridDimension) % gridDimension]; // West

                    Particle lbestParticle = neighbours.OrderBy(x => x.ebest).First();
                    double[] lbest = lbestParticle.pbest;
                    double elbest = lbestParticle.ebest;

                    if (newError < gebest)
                    {
                        gebest = newError;
                        particle.position.CopyTo(gbest, 0);
                    }

                    for (int k = 0; k < particle.velocity.Length; k++)
                    {
                        r1 = rnd.NextDouble();
                        r2 = rnd.NextDouble();

                        particle.velocity[k] = (w * particle.velocity[k]) +
                                               (c1 * r1 * (particle.pbest[k] - particle.position[k])) +
                                               (c2 * r2 * (lbest[k] - particle.position[k]));

                        particle.position[k] += particle.velocity[k];
                    }
                }
            }

            //Reorders the swarm based on position and resets the grid
            if (iteration % mod == 0)
            {
                OrderSwarmBasedOnPosition(swarm);

                for (int p = 0; p < swarm.Length; p++)
                {
                    grid[p / gridDimension, p % gridDimension] = swarm[p];
                }
            }

            iteration++;
            stopwatch.Stop();
            timeElapsed = stopwatch.Elapsed;

            if (timeElapsed > counter)
            {
                this.SetWeights(gbest);
                double accuracy = TestAccuracy(testData);
                

                string elapsedTime = String.Format("{0:00}:{1:00}:{2:00}.{3:00}",
                timeElapsed.Hours, timeElapsed.Minutes, timeElapsed.Seconds,
                timeElapsed.Milliseconds / 10);



                accuracyList.Add(accuracy);
                timeList.Add(elapsedTime);

                if (accuracyCounter > 0)
                {
                    if (accuracyList[accuracyCounter] < accuracyList[accuracyCounter - 1])
                    {
                        accuracyList[accuracyCounter] = accuracyList[accuracyCounter - 1];
                    }
                }

                accuracyCounter++;

                counter = counter.Add(interval);
            }
        }

        this.SetWeights(gbest);
        double[] result = new double[totalWeights];
        Array.Copy(gbest, result, result.Length);
        SaveToExcel(timeList, accuracyList, top);
        return result;
    }

    public double[] TrainRandomVonNeumann(double[][] trainingData, int numParticles, int maxIterations, double exitError, int gridDimension, int mod, TimeSpan exitTime, double[][] testData, TimeSpan interval)
    {
        int totalWeights = (this.numInput * this.numHidden) + (this.numHidden * this.numOutput) +
        this.numHidden + this.numOutput;

        System.Diagnostics.Stopwatch stopwatch = System.Diagnostics.Stopwatch.StartNew();
        stopwatch.Stop();
        TimeSpan timeElapsed = stopwatch.Elapsed;

        TimeSpan counter = new TimeSpan(00, 00, 00);

        Random rnd = new Random();

        int iteration = 0;
        double minX = -10;
        double maxX = 10;
        double w = 0.4;
        double c1 = 1.49445;
        double c2 = 1.49445;
        double r1, r2;
        String top = "RVN";
        var accuracyList = new List<double>();
        var timeList = new List<string>();
        int accuracyCounter = 0;

        // Create grid for von Neumann topology
        Particle[,] grid = new Particle[gridDimension, gridDimension];
        Particle[] swarm = new Particle[numParticles];

        double[] gbest = new double[totalWeights]; //best global position
        double gebest = double.MaxValue; //best global error

        // Initialize particles and assign to grid
        for (int p = 0; p < swarm.Length; p++)
        {
            double[] startingPos = new double[totalWeights];
            for (int j = 0; j < startingPos.Length; j++)
            {
                startingPos[j] += (rnd.NextDouble() * 2 - 1) * 0.1;
            }

            double error = MeanSqrError(trainingData, startingPos);

            double[] startingVelocity = new double[totalWeights];

            for (int j = 0; j < startingVelocity.Length; j++)
            {
                double lo = 0.1 * minX;
                double hi = 0.1 * maxX;
                startingVelocity[j] = (hi - lo) * rnd.NextDouble() + lo;
            }


            swarm[p] = new Particle(startingPos, startingVelocity, startingPos, error, error);
            grid[p / gridDimension, p % gridDimension] = swarm[p];

            if (swarm[p].error < gebest)
            {
                gebest = swarm[p].error;
                swarm[p].position.CopyTo(gbest, 0);
            }
        }

        while (timeElapsed < exitTime)
        {
            stopwatch.Start();
            for (int i = 0; i < gridDimension; i++)
            {
                for (int j = 0; j < gridDimension; j++)
                {
                    Particle particle = grid[i, j];
                    double newError = MeanSqrError(trainingData, particle.position);

                    if (newError < particle.ebest)
                    {
                        particle.ebest = newError;
                        particle.position.CopyTo(particle.pbest, 0);
                    }

                    // Compute local best for Von Neumann neighbourhood
                    Particle[] neighbours = new Particle[4];
                    neighbours[0] = grid[(i - 1 + gridDimension) % gridDimension, j]; // North
                    neighbours[1] = grid[i, (j + 1) % gridDimension]; // East
                    neighbours[2] = grid[(i + 1) % gridDimension, j]; // South
                    neighbours[3] = grid[i, (j - 1 + gridDimension) % gridDimension]; // West

                    Particle lbestParticle = neighbours.OrderBy(x => x.ebest).First();
                    double[] lbest = lbestParticle.pbest;
                    double elbest = lbestParticle.ebest;

                    if (newError < gebest)
                    {
                        gebest = newError;
                        particle.position.CopyTo(gbest, 0);
                    }

                    for (int k = 0; k < particle.velocity.Length; k++)
                    {
                        r1 = rnd.NextDouble();
                        r2 = rnd.NextDouble();

                        particle.velocity[k] = (w * particle.velocity[k]) +
                                               (c1 * r1 * (particle.pbest[k] - particle.position[k])) +
                                               (c2 * r2 * (lbest[k] - particle.position[k]));

                        particle.position[k] += particle.velocity[k];
                    }
                }
            }

            //Reorders the swarm based on position and resets the grid
            if (iteration % mod == 0)
            {
                Shuffle(swarm);

                for (int p = 0; p < swarm.Length; p++)
                {
                    grid[p / gridDimension, p % gridDimension] = swarm[p];
                }
            }

            iteration++;
            stopwatch.Stop();
            timeElapsed = stopwatch.Elapsed;

            if (timeElapsed > counter)
            {
                this.SetWeights(gbest);
                double accuracy = TestAccuracy(testData);
                

                string elapsedTime = String.Format("{0:00}:{1:00}:{2:00}.{3:00}",
                timeElapsed.Hours, timeElapsed.Minutes, timeElapsed.Seconds,
                timeElapsed.Milliseconds / 10);



                accuracyList.Add(accuracy);
                timeList.Add(elapsedTime);

                if (accuracyCounter > 0)
                {
                    if (accuracyList[accuracyCounter] < accuracyList[accuracyCounter - 1])
                    {
                        accuracyList[accuracyCounter] = accuracyList[accuracyCounter - 1];
                    }
                }

                accuracyCounter++;

                counter = counter.Add(interval);
            }
        }

        this.SetWeights(gbest);
        double[] result = new double[totalWeights];
        Array.Copy(gbest, result, result.Length);
        SaveToExcel(timeList, accuracyList, top);
        return result;
    }

    //Orders a swarm based on their euclidean distance
    public void OrderSwarmBasedOnPosition(Particle[] swarm)
    {
        Array.Sort(swarm, (a, b) =>
        {
            double distA = EuclideanDistance(a.position, new double[a.position.Length]);
            double distB = EuclideanDistance(b.position, new double[b.position.Length]);

            return distA.CompareTo(distB);
        });
    }

    //Calculate Euclidean Distance between 2 particles of a swarm
    public double EuclideanDistance(double[] position1, double[] position2)
    {
        double sum = 0.0;

        for (int i = 0; i < position1.Length; i++)
        {
            sum += Math.Pow(position1[i] - position2[i], 2);
        }

        return Math.Sqrt(sum);
    }

    private void Shuffle(Particle[] swarm)
    {
        Random rand = new Random();
        for (int i = swarm.Length - 1; i > 0; i--)
        {
            int j = rand.Next(i + 1);
            // Swap swarm[i] with swarm[j]
            Particle temp = swarm[i];
            swarm[i] = swarm[j];
            swarm[j] = temp;
        }
    }


    // Set weights and biases based on a single array
    public void SetWeights(double[] weights)
    {
        int k = 0;

        // Set weights between input layer and hidden layer
        for (int i = 0; i < numInput; i++)
        {
            for (int j = 0; j < numHidden; j++)
            {
                ihWeights[i][j] = weights[k++];
            }
        }

        // Set biases for hidden layer
        for (int i = 0; i < numHidden; i++)
        {
            hBiases[i] = weights[k++];
        }

        // Set weights between hidden layer and output layer
        for (int i = 0; i < numHidden; i++)
        {
            for (int j = 0; j < numOutput; j++)
            {
                hoWeights[i][j] = weights[k++];
            }
        }

        // Set biases for output layer
        for (int i = 0; i < numOutput; i++)
        {
            oBiases[i] = weights[k++];
        }
    }

    // Flatten all weights and biases into a single array and calls the array
    public double[] GetWeights()
    {
        int totalWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
        double[] result = new double[totalWeights];
        int k = 0;

        for (int i = 0; i < ihWeights.Length; i++)
        {
            for (int j = 0; j < ihWeights[0].Length; j++)
            {
                result[k++] = ihWeights[i][j];
            }
        }

        for (int i = 0; i < hBiases.Length; i++)
        {
            result[k++] = hBiases[i];
        }

        for (int i = 0; i < hoWeights.Length; i++)
        {
            for (int j = 0; j < hoWeights[0].Length; j++)
            {
                result[k++] = hoWeights[i][j];
            }
        }

        for (int i = 0; i < oBiases.Length; i++)
        {
            result[k++] = oBiases[i];
        }

        return result;
    }

    //calculates the average error of the network across a dataset
    private double MeanSqrError(double[][] trainingData, double[] weights)
    {
        this.SetWeights(weights);

        double[] xValues = new double[numInput];
        double[] tValues = new double[numOutput];
        double sumSquaredError = 0.0;
        for (int i = 0; i < trainingData.Length; i++)
        {
            if (trainingData[i].Length < numInput + numOutput)
                throw new Exception($"Training data at index {i} does not have enough values. Expected at least {numInput + numOutput} but got {trainingData[i].Length}.");

            Array.Copy(trainingData[i], xValues, numInput);
            Array.Copy(trainingData[i], numInput, tValues, 0, numOutput);
            double[] yValues = this.Feedforward(xValues);
            for (int j = 0; j < yValues.Length; j++)
                sumSquaredError += ((yValues[j] - tValues[j]) * (yValues[j] - tValues[j]));
        }
        return sumSquaredError / trainingData.Length;
    }

    //Does the same as MeanSqrError but upside down
    public double TestAccuracy(double[][] testData)
    {
        int correctPredictions = 0;

        for (int i = 0; i < testData.Length; i++)
        {
            int expectedOutput = (int)testData[i][0];
            double[] inputs = testData[i].Skip(1).ToArray();

            double[] outputs = Feedforward(inputs);

            
            int predictedClass = Array.IndexOf(outputs, outputs.Max());

            if (predictedClass == expectedOutput) 
            {
                correctPredictions++;
            }

        }

        return ((double)correctPredictions / testData.Length) * 100;
    }

    //saves 2 lists into an excel document
    void SaveToExcel(List<string> elapsedTimeList, List<double> accuracyList, string topography)
    {
        string excelFilePath = "C:/Users/andre/source/repos/NeuralNetworkPSOTraining/NeuralNetworkPSOTraining/Data/Results.xlsx";

        FileInfo fileInfo = new FileInfo(excelFilePath);

        ExcelPackage.LicenseContext = LicenseContext.NonCommercial;

        using (ExcelPackage package = new ExcelPackage(fileInfo))
        {
            ExcelWorksheet worksheet = package.Workbook.Worksheets.FirstOrDefault(ws => ws.Name == topography);

            if (worksheet == null)
                worksheet = package.Workbook.Worksheets.Add(topography);

            int rowCount = worksheet.Dimension?.End.Row ?? 0;

            for (int i = 0; i < Math.Max(elapsedTimeList.Count, accuracyList.Count); i++)
            {
                if (i < elapsedTimeList.Count)
                    worksheet.Cells[rowCount + 1 + i, 1].Value = elapsedTimeList[i];

                if (i < accuracyList.Count)
                    worksheet.Cells[rowCount + 1 + i, 2].Value = accuracyList[i];
            }

            package.Save();
        }
    }

}
