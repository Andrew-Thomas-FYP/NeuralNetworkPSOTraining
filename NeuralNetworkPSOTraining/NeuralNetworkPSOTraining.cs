using OfficeOpenXml;
using System;
using System.ComponentModel;
using System.Data;
using System.IO;
class Program
{
    static void Main(string[] args)
    {
        string dataset = "phone";
        int runs = 1;
        int numParticles = 16; //must be a square number for vonNeumann to work
        int maxIterations = 1000000000;
        double exitError = 0.00; //Percentage Error where training stops
        int numInput = GetInputNeurons(dataset); //Define the number of input nodes
        int numHidden = 20; // Define the number of hidden layer nodes
        int numOutput = GetOutputNeurons(dataset); // Define the number of output classes
        int mod = 20; // Frequency that the dynamic topologies change topology in number of iterations
        int gridDimension = (int)Math.Sqrt(numParticles);

        TimeSpan exitTime = new TimeSpan(00,00,30);
        TimeSpan interval = new TimeSpan(00,00,01);

        //Arrays for storing all star topology runs accuracy and time
        TimeSpan[] starAllTime = new TimeSpan[runs];
        double[] starAllAccuracy = new double[runs];

        //Arrays for storing all circle topology runs accuracy and time
        TimeSpan[] circleAllTime = new TimeSpan[runs];
        double[] circleAllAccuracy = new double[runs];

        //Arrays for storing all euclidean circle topology runs accuracy and time
        TimeSpan[] euclideanCircleAllTime = new TimeSpan[runs];
        double[] euclideanCircleAllAccuracy = new double[runs];

        //Arrays for storing all random circle topology runs accuracy and time
        TimeSpan[] randomCircleAllTime = new TimeSpan[runs];
        double[] randomCircleAllAccuracy = new double[runs];

        //Arrays for storing all Von Neumann topology runs accuracy and time
        TimeSpan[] vonNeumannAllTime = new TimeSpan[runs];
        double[] vonNeumannAllAccuracy = new double[runs];

        //Arrays for storing all euclidean Von Neumann topology runs accuracy and time
        TimeSpan[] euclideanVonNeumannAllTime = new TimeSpan[runs];
        double[] euclideanVonNeumannAllAccuracy = new double[runs];

        //Arrays for storing all random Von Neumann topology runs accuracy and time
        TimeSpan[] randomVonNeumannAllTime = new TimeSpan[runs];
        double[] randomVonNeumannAllAccuracy = new double[runs];

        // Load data from file
        string dataName = Path.Combine("C:/Users/andre/source/repos/NeuralNetworkPSOTraining/NeuralNetworkPSOTraining/Data/" + dataset + ".csv");
        var data = File.ReadLines(dataName)
            .Skip(1)  // Skip the header row
            .Select(line => line.Split(',').Select(double.Parse).ToArray())
            .ToArray();

        // Determine train-test split
        int trainSize = (int)(0.8 * data.Length); // % of data used for training
        int testSize = data.Length - trainSize; // rest for testing

        // A set of runs using the star topology training method
        for (int i = 0; i < runs; i++)
        {
            System.Diagnostics.Stopwatch stopwatch = System.Diagnostics.Stopwatch.StartNew();
            Console.WriteLine("star run: {0}", i + 1);


            // Prepare training and testing datasets
            var trainingData = new double[trainSize][];
            var testData = new double[testSize][];

            Random rnd = new Random();
            data = data.OrderBy(x => rnd.Next()).ToArray(); // Shuffle the data

            Array.Copy(data, 0, trainingData, 0, trainSize);
            Array.Copy(data, trainSize, testData, 0, testSize);

            // Initialize neural network

            NeuralNetwork_v2 nn = new NeuralNetwork_v2(numInput, numHidden, numOutput);

            //onehot the data
            int[] labels = trainingData.Select(instance => (int)instance[0]).ToArray();
            double[][] oneHotLabels = OneHotEncode(labels, numOutput);
            double[][] inputs = trainingData.Select(instance => instance.Skip(1).ToArray()).ToArray();
            double[][] newTrainingData = new double[inputs.Length][];

            for (int j = 0; j < inputs.Length; j++)
            {
                newTrainingData[j] = inputs[j].Concat(oneHotLabels[j]).ToArray();
            }

            //Train the neural network
            double[] bestWeights = nn.TrainStar(newTrainingData, numParticles, maxIterations, exitError, exitTime, testData, interval);

            // Test the accuracy of the trained neural network
            nn.SetWeights(bestWeights);
            double accuracy = nn.TestAccuracy(testData);
            Console.WriteLine("Accuracy: {0}%", accuracy);

            stopwatch.Stop();
            TimeSpan ts = stopwatch.Elapsed;

            // Format and display the TimeSpan value.
            string elapsedTime = String.Format("{0:00}:{1:00}:{2:00}.{3:00}",
                ts.Hours, ts.Minutes, ts.Seconds,
                ts.Milliseconds / 10);
            Console.WriteLine("RunTime " + elapsedTime);
            Console.WriteLine();

            starAllTime[i] = ts;
            starAllAccuracy[i] = accuracy;
        }

        double starAverageAccuracy = GetAverage(starAllAccuracy);
        double starStndDev = StandardDeviation(starAllAccuracy);

        double starAverageTicks = starAllTime.Select(time => time.Ticks).Average();
        TimeSpan starAverageTime = TimeSpan.FromTicks((long)starAverageTicks);

        double starTicksStndDev = TimeSpanStandardDeviation(starAllTime);
        TimeSpan starTimeStndDev = TimeSpan.FromTicks((long)starTicksStndDev);



        // A set of runs using the circle topology training method
        for (int i = 0; i < runs; i++)
        {
            System.Diagnostics.Stopwatch stopwatch = System.Diagnostics.Stopwatch.StartNew();
            Console.WriteLine("circle run: {0}", i + 1);

            // Prepare training and testing datasets
            var trainingData = new double[trainSize][];
            var testData = new double[testSize][];

            Random rnd = new Random();
            data = data.OrderBy(x => rnd.Next()).ToArray(); // Shuffle the data

            Array.Copy(data, 0, trainingData, 0, trainSize);
            Array.Copy(data, trainSize, testData, 0, testSize);

            // Initialize neural network

            NeuralNetwork_v2 nn = new NeuralNetwork_v2(numInput, numHidden, numOutput);

            //onehot the data
            int[] labels = trainingData.Select(instance => (int)instance[0]).ToArray();
            double[][] oneHotLabels = OneHotEncode(labels, numOutput);
            double[][] inputs = trainingData.Select(instance => instance.Skip(1).ToArray()).ToArray();
            double[][] newTrainingData = new double[inputs.Length][];

            for (int j = 0; j < inputs.Length; j++)
            {
                newTrainingData[j] = inputs[j].Concat(oneHotLabels[j]).ToArray();
            }

            //Train the neural network
            double[] bestWeights = nn.TrainCircle(newTrainingData, numParticles, maxIterations, exitError, exitTime, testData, interval);

            // Test the accuracy of the trained neural network
            nn.SetWeights(bestWeights);
            double accuracy = nn.TestAccuracy(testData);
            Console.WriteLine("Accuracy: {0}%", accuracy);

            stopwatch.Stop();
            TimeSpan ts = stopwatch.Elapsed;

            // Format and display the TimeSpan value.
            string elapsedTime = String.Format("{0:00}:{1:00}:{2:00}.{3:00}",
                ts.Hours, ts.Minutes, ts.Seconds,
                ts.Milliseconds / 10);
            Console.WriteLine("RunTime " + elapsedTime);
            Console.WriteLine();

            circleAllTime[i] = ts;
            circleAllAccuracy[i] = accuracy;
        }

        double circleAverageAccuracy = GetAverage(circleAllAccuracy);
        double circleStndDev = StandardDeviation(circleAllAccuracy);

        double circleAverageTicks = circleAllTime.Select(time => time.Ticks).Average();
        TimeSpan circleAverageTime = TimeSpan.FromTicks((long)circleAverageTicks);

        double circleTicksStndDev = TimeSpanStandardDeviation(circleAllTime);
        TimeSpan circleTimeStndDev = TimeSpan.FromTicks((long)circleTicksStndDev);

        //A set of runs using VonNeumann Topology
        for (int i = 0; i < runs; i++)
        {
            System.Diagnostics.Stopwatch stopwatch = System.Diagnostics.Stopwatch.StartNew();
            Console.WriteLine("Von Neumann run: {0}", i + 1);


            // Prepare training and testing datasets
            var trainingData = new double[trainSize][];
            var testData = new double[testSize][];

            Random rnd = new Random();
            data = data.OrderBy(x => rnd.Next()).ToArray(); // Shuffle the data

            Array.Copy(data, 0, trainingData, 0, trainSize);
            Array.Copy(data, trainSize, testData, 0, testSize);

            // Initialize neural network

            NeuralNetwork_v2 nn = new NeuralNetwork_v2(numInput, numHidden, numOutput);

            //onehot the data
            int[] labels = trainingData.Select(instance => (int)instance[0]).ToArray();
            double[][] oneHotLabels = OneHotEncode(labels, numOutput);
            double[][] inputs = trainingData.Select(instance => instance.Skip(1).ToArray()).ToArray();
            double[][] newTrainingData = new double[inputs.Length][];

            for (int j = 0; j < inputs.Length; j++)
            {
                newTrainingData[j] = inputs[j].Concat(oneHotLabels[j]).ToArray();
            }

            //Train the neural network
            double[] bestWeights = nn.TrainVonNeumann(newTrainingData, numParticles, maxIterations, exitError, gridDimension, exitTime, testData, interval);

            // Test the accuracy of the trained neural network
            nn.SetWeights(bestWeights);
            double accuracy = nn.TestAccuracy(testData);
            Console.WriteLine("Accuracy: {0}%", accuracy);

            stopwatch.Stop();
            TimeSpan ts = stopwatch.Elapsed;

            // Format and display the TimeSpan value.
            string elapsedTime = String.Format("{0:00}:{1:00}:{2:00}.{3:00}",
                ts.Hours, ts.Minutes, ts.Seconds,
                ts.Milliseconds / 10);
            Console.WriteLine("RunTime " + elapsedTime);
            Console.WriteLine();

            vonNeumannAllTime[i] = ts;
            vonNeumannAllAccuracy[i] = accuracy;
        }

        double vonNeumannAverageAccuracy = GetAverage(vonNeumannAllAccuracy);
        double vonNeumannStndDev = StandardDeviation(vonNeumannAllAccuracy);

        double vonNeumannAverageTicks = vonNeumannAllTime.Select(time => time.Ticks).Average();
        TimeSpan vonNeumannAverageTime = TimeSpan.FromTicks((long)vonNeumannAverageTicks);

        double vonNeumannTicksStndDev = TimeSpanStandardDeviation(vonNeumannAllTime);
        TimeSpan vonNeumannTimeStndDev = TimeSpan.FromTicks((long)vonNeumannTicksStndDev);

        // A set of runs using the Euclidean Circle topology training method
        for (int i = 0; i < runs; i++)
        {
            System.Diagnostics.Stopwatch stopwatch = System.Diagnostics.Stopwatch.StartNew();
            Console.WriteLine("Euclidean Circle run: {0}", i + 1);

            // Prepare training and testing datasets
            var trainingData = new double[trainSize][];
            var testData = new double[testSize][];

            Random rnd = new Random();
            data = data.OrderBy(x => rnd.Next()).ToArray(); // Shuffle the data

            Array.Copy(data, 0, trainingData, 0, trainSize);
            Array.Copy(data, trainSize, testData, 0, testSize);

            // Initialize neural network

            NeuralNetwork_v2 nn = new NeuralNetwork_v2(numInput, numHidden, numOutput);

            //onehot the data
            int[] labels = trainingData.Select(instance => (int)instance[0]).ToArray();
            double[][] oneHotLabels = OneHotEncode(labels, numOutput);
            double[][] inputs = trainingData.Select(instance => instance.Skip(1).ToArray()).ToArray();
            double[][] newTrainingData = new double[inputs.Length][];

            for (int j = 0; j < inputs.Length; j++)
            {
                newTrainingData[j] = inputs[j].Concat(oneHotLabels[j]).ToArray();
            }

            //Train the neural network
            double[] bestWeights = nn.TrainEuclidianCircle(newTrainingData, numParticles, maxIterations, exitError, mod, exitTime, testData, interval);

            // Test the accuracy of the trained neural network
            nn.SetWeights(bestWeights);
            double accuracy = nn.TestAccuracy(testData);
            Console.WriteLine("Accuracy: {0}%", accuracy);

            stopwatch.Stop();
            TimeSpan ts = stopwatch.Elapsed;

            // Format and display the TimeSpan value.
            string elapsedTime = String.Format("{0:00}:{1:00}:{2:00}.{3:00}",
                ts.Hours, ts.Minutes, ts.Seconds,
                ts.Milliseconds / 10);
            Console.WriteLine("RunTime " + elapsedTime);
            Console.WriteLine();

            euclideanCircleAllTime[i] = ts;
            euclideanCircleAllAccuracy[i] = accuracy;
        }

        double euclideanCircleAverageAccuracy = GetAverage(euclideanCircleAllAccuracy);
        double euclideanCircleStndDev = StandardDeviation(euclideanCircleAllAccuracy);

        double euclideanCircleAverageTicks = euclideanCircleAllTime.Select(time => time.Ticks).Average();
        TimeSpan euclideanCircleAverageTime = TimeSpan.FromTicks((long)euclideanCircleAverageTicks);

        double euclideanCircleTicksStndDev = TimeSpanStandardDeviation(euclideanCircleAllTime);
        TimeSpan euclideanCircleTimeStndDev = TimeSpan.FromTicks((long)euclideanCircleTicksStndDev);

        //A set of runs using euclidean Von Neumann Topology
        for (int i = 0; i < runs; i++)
        {
            System.Diagnostics.Stopwatch stopwatch = System.Diagnostics.Stopwatch.StartNew();
            Console.WriteLine("Euclidean Von Neumann run: {0}", i + 1);


            // Prepare training and testing datasets
            var trainingData = new double[trainSize][];
            var testData = new double[testSize][];

            Random rnd = new Random();
            data = data.OrderBy(x => rnd.Next()).ToArray(); // Shuffle the data

            Array.Copy(data, 0, trainingData, 0, trainSize);
            Array.Copy(data, trainSize, testData, 0, testSize);

            // Initialize neural network

            NeuralNetwork_v2 nn = new NeuralNetwork_v2(numInput, numHidden, numOutput);

            //onehot the data
            int[] labels = trainingData.Select(instance => (int)instance[0]).ToArray();
            double[][] oneHotLabels = OneHotEncode(labels, numOutput);
            double[][] inputs = trainingData.Select(instance => instance.Skip(1).ToArray()).ToArray();
            double[][] newTrainingData = new double[inputs.Length][];

            for (int j = 0; j < inputs.Length; j++)
            {
                newTrainingData[j] = inputs[j].Concat(oneHotLabels[j]).ToArray();
            }

            //Train the neural network
            double[] bestWeights = nn.TrainEuclideanVonNeumann(newTrainingData, numParticles, maxIterations, exitError, gridDimension, mod, exitTime, testData, interval);

            // Test the accuracy of the trained neural network
            nn.SetWeights(bestWeights);
            double accuracy = nn.TestAccuracy(testData);
            Console.WriteLine("Accuracy: {0}%", accuracy);

            stopwatch.Stop();
            TimeSpan ts = stopwatch.Elapsed;

            // Format and display the TimeSpan value.
            string elapsedTime = String.Format("{0:00}:{1:00}:{2:00}.{3:00}",
                ts.Hours, ts.Minutes, ts.Seconds,
                ts.Milliseconds / 10);
            Console.WriteLine("RunTime " + elapsedTime);
            Console.WriteLine();

            euclideanVonNeumannAllTime[i] = ts;
            euclideanVonNeumannAllAccuracy[i] = accuracy;
        }

        double euclideanVonNeumannAverageAccuracy = GetAverage(euclideanVonNeumannAllAccuracy);
        double euclideanVonNeumannStndDev = StandardDeviation(euclideanVonNeumannAllAccuracy);

        double euclideanVonNeumannAverageTicks = euclideanVonNeumannAllTime.Select(time => time.Ticks).Average();
        TimeSpan euclideanVonNeumannAverageTime = TimeSpan.FromTicks((long)euclideanVonNeumannAverageTicks);

        double euclideanVonNeumannTicksStndDev = TimeSpanStandardDeviation(euclideanVonNeumannAllTime);
        TimeSpan euclideanVonNeumannTimeStndDev = TimeSpan.FromTicks((long)euclideanVonNeumannTicksStndDev);

        // A set of runs using the Random Circle topology training method
        for (int i = 0; i < runs; i++)
        {
            System.Diagnostics.Stopwatch stopwatch = System.Diagnostics.Stopwatch.StartNew();
            Console.WriteLine("Random Circle run: {0}", i + 1);

            // Prepare training and testing datasets
            var trainingData = new double[trainSize][];
            var testData = new double[testSize][];

            Random rnd = new Random();
            data = data.OrderBy(x => rnd.Next()).ToArray(); // Shuffle the data

            Array.Copy(data, 0, trainingData, 0, trainSize);
            Array.Copy(data, trainSize, testData, 0, testSize);

            // Initialize neural network

            NeuralNetwork_v2 nn = new NeuralNetwork_v2(numInput, numHidden, numOutput);

            //onehot the data
            int[] labels = trainingData.Select(instance => (int)instance[0]).ToArray();
            double[][] oneHotLabels = OneHotEncode(labels, numOutput);
            double[][] inputs = trainingData.Select(instance => instance.Skip(1).ToArray()).ToArray();
            double[][] newTrainingData = new double[inputs.Length][];

            for (int j = 0; j < inputs.Length; j++)
            {
                newTrainingData[j] = inputs[j].Concat(oneHotLabels[j]).ToArray();
            }

            //Train the neural network
            double[] bestWeights = nn.TrainRandomCircle(newTrainingData, numParticles, maxIterations, exitError, exitTime, testData, interval);

            // Test the accuracy of the trained neural network
            nn.SetWeights(bestWeights);
            double accuracy = nn.TestAccuracy(testData);
            Console.WriteLine("Accuracy: {0}%", accuracy);

            stopwatch.Stop();
            TimeSpan ts = stopwatch.Elapsed;

            // Format and display the TimeSpan value.
            string elapsedTime = String.Format("{0:00}:{1:00}:{2:00}.{3:00}",
                ts.Hours, ts.Minutes, ts.Seconds,
                ts.Milliseconds / 10);
            Console.WriteLine("RunTime " + elapsedTime);
            Console.WriteLine();

            randomCircleAllTime[i] = ts;
            randomCircleAllAccuracy[i] = accuracy;
        }

        double randomCircleAverageAccuracy = GetAverage(randomCircleAllAccuracy);
        double randomCircleStndDev = StandardDeviation(randomCircleAllAccuracy);

        double randomCircleAverageTicks = randomCircleAllTime.Select(time => time.Ticks).Average();
        TimeSpan randomCircleAverageTime = TimeSpan.FromTicks((long)randomCircleAverageTicks);

        double randomCircleTicksStndDev = TimeSpanStandardDeviation(randomCircleAllTime);
        TimeSpan randomCircleTimeStndDev = TimeSpan.FromTicks((long)randomCircleTicksStndDev);

        //A set of runs using Random Von Neumann Topology
        for (int i = 0; i < runs; i++)
        {
            System.Diagnostics.Stopwatch stopwatch = System.Diagnostics.Stopwatch.StartNew();
            Console.WriteLine("Random Von Neumann run: {0}", i + 1);


            // Prepare training and testing datasets
            var trainingData = new double[trainSize][];
            var testData = new double[testSize][];

            Random rnd = new Random();
            data = data.OrderBy(x => rnd.Next()).ToArray(); // Shuffle the data

            Array.Copy(data, 0, trainingData, 0, trainSize);
            Array.Copy(data, trainSize, testData, 0, testSize);

            // Initialize neural network

            NeuralNetwork_v2 nn = new NeuralNetwork_v2(numInput, numHidden, numOutput);

            //onehot the data
            int[] labels = trainingData.Select(instance => (int)instance[0]).ToArray();
            double[][] oneHotLabels = OneHotEncode(labels, numOutput);
            double[][] inputs = trainingData.Select(instance => instance.Skip(1).ToArray()).ToArray();
            double[][] newTrainingData = new double[inputs.Length][];

            for (int j = 0; j < inputs.Length; j++)
            {
                newTrainingData[j] = inputs[j].Concat(oneHotLabels[j]).ToArray();
            }

            //Train the neural network
            double[] bestWeights = nn.TrainRandomVonNeumann(newTrainingData, numParticles, maxIterations, exitError, gridDimension, mod, exitTime, testData, interval);

            // Test the accuracy of the trained neural network
            nn.SetWeights(bestWeights);
            double accuracy = nn.TestAccuracy(testData);
            Console.WriteLine("Accuracy: {0}%", accuracy);

            stopwatch.Stop();
            TimeSpan ts = stopwatch.Elapsed;

            // Format and display the TimeSpan value.
            string elapsedTime = String.Format("{0:00}:{1:00}:{2:00}.{3:00}",
                ts.Hours, ts.Minutes, ts.Seconds,
                ts.Milliseconds / 10);
            Console.WriteLine("RunTime " + elapsedTime);
            Console.WriteLine();

            randomVonNeumannAllTime[i] = ts;
            randomVonNeumannAllAccuracy[i] = accuracy;
        }

        double randomVonNeumannAverageAccuracy = GetAverage(randomVonNeumannAllAccuracy);
        double randomVonNeumannStndDev = StandardDeviation(randomVonNeumannAllAccuracy);

        double randomVonNeumannAverageTicks = randomVonNeumannAllTime.Select(time => time.Ticks).Average();
        TimeSpan randomVonNeumannAverageTime = TimeSpan.FromTicks((long)randomVonNeumannAverageTicks);

        double randomVonNeumannTicksStndDev = TimeSpanStandardDeviation(randomVonNeumannAllTime);
        TimeSpan randomVonNeumannTimeStndDev = TimeSpan.FromTicks((long)randomVonNeumannTicksStndDev);

        //prints the average results of all the different runs
        Console.WriteLine();
        Console.WriteLine("Uses dataset: {0}", dataset);
        Console.WriteLine();
        Console.WriteLine("Number of Input Neurons: {0}", numInput);
        Console.WriteLine("Number of Hidden Neurons: {0}", numHidden);
        Console.WriteLine("Number of Output Neurons: {0}", numOutput);
        Console.WriteLine("Number of Particles: {0}", numParticles);
        Console.WriteLine("Maximum Number of Iterations Per Run: {0}", maxIterations);
        Console.WriteLine("Target accuracy per run: {0}%", ((1 - exitError) * 100));
        Console.WriteLine("Frequency of Euclidean Topologies Changing: {0}", mod);
        Console.WriteLine();
        Console.WriteLine("These results were taken from {0} runs of the program", runs);
        Console.WriteLine();
        Console.WriteLine("Average star accuracy is: {0}%", starAverageAccuracy);
        Console.WriteLine("Star standard deviation is: {0}", starStndDev);
        Console.WriteLine("Average star runtime is: {0:hh\\:mm\\:ss\\.fff}", starAverageTime);
        Console.WriteLine("Star runtime standard deviation is: {0:hh\\:mm\\:ss\\.fff}", starTimeStndDev);
        Console.WriteLine();
        Console.WriteLine("Average static circle accuracy is: {0}%", circleAverageAccuracy);
        Console.WriteLine("Circle standard deviation is: {0}", circleStndDev);
        Console.WriteLine("Average static circle runtime is: {0:hh\\:mm\\:ss\\.fff}", circleAverageTime);
        Console.WriteLine("Circle runtime standard deviation is: {0:hh\\:mm\\:ss\\.fff}", circleTimeStndDev);
        Console.WriteLine();
        Console.WriteLine("Average Von Neumann accuracy is: {0}%", vonNeumannAverageAccuracy);
        Console.WriteLine("Von Neumann standard deviation is: {0}", vonNeumannStndDev);
        Console.WriteLine("Average Von Neumann runtime is: {0:hh\\:mm\\:ss\\.fff}", vonNeumannAverageTime);
        Console.WriteLine("Von Neumann runtime standard deviation is: {0:hh\\:mm\\:ss\\.fff}", vonNeumannTimeStndDev);
        Console.WriteLine();
        Console.WriteLine("Average euclidean circle accuracy is: {0}%", euclideanCircleAverageAccuracy);
        Console.WriteLine("Euclidean circle standard deviation is: {0}", euclideanCircleStndDev);
        Console.WriteLine("Average euclidean circle runtime is: {0:hh\\:mm\\:ss\\.fff}", euclideanCircleAverageTime);
        Console.WriteLine("Euclidean circle runtime standard deviation is: {0:hh\\:mm\\:ss\\.fff}", euclideanCircleTimeStndDev);
        Console.WriteLine();
        Console.WriteLine("Average euclidean Von Neumann accuracy is: {0}%", euclideanVonNeumannAverageAccuracy);
        Console.WriteLine("Euclidean Von Neumann standard deviation is: {0}", euclideanVonNeumannStndDev);
        Console.WriteLine("Average euclidean Von Neumann runtime is: {0:hh\\:mm\\:ss\\.fff}", euclideanVonNeumannAverageTime);
        Console.WriteLine("Euclidean von neumann runtime standard deviation is: {0:hh\\:mm\\:ss\\.fff}", euclideanVonNeumannTimeStndDev);
        Console.WriteLine();
        Console.WriteLine("Average random circle accuracy is: {0}%", randomCircleAverageAccuracy);
        Console.WriteLine("Random circle standard deviation is: {0}", randomCircleStndDev);
        Console.WriteLine("Average random circle runtime is: {0:hh\\:mm\\:ss\\.fff}", randomCircleAverageTime);
        Console.WriteLine("Random circle runtime standard deviation is: {0:hh\\:mm\\:ss\\.fff}", randomCircleTimeStndDev);
        Console.WriteLine();
        Console.WriteLine("Average random Von Neumann accuracy is: {0}%", randomVonNeumannAverageAccuracy);
        Console.WriteLine("Random Von Neumann standard deviation is: {0}", randomVonNeumannStndDev);
        Console.WriteLine("Average random Von Neumann runtime is: {0:hh\\:mm\\:ss\\.fff}", randomVonNeumannAverageTime);
        Console.WriteLine("Random von neumann runtime standard deviation is: {0:hh\\:mm\\:ss\\.fff}", randomVonNeumannTimeStndDev);

        // Format the average times and standard deviations into a more readable form for Excel
        string starAverageTimeFormatted = starAverageTime.ToString(@"hh\:mm\:ss\.fff");
        string circleAverageTimeFormatted = circleAverageTime.ToString(@"hh\:mm\:ss\.fff");
        string vonNeumannAverageTimeFormatted = vonNeumannAverageTime.ToString(@"hh\:mm\:ss\.fff");
        string euclideanCircleAverageTimeFormatted = euclideanCircleAverageTime.ToString(@"hh\:mm\:ss\.fff");
        string euclideanVonNeumannAverageTimeFormatted = euclideanVonNeumannAverageTime.ToString(@"hh\:mm\:ss\.fff");
        string randomCircleAverageTimeFormatted = randomCircleAverageTime.ToString(@"hh\:mm\:ss\.fff");
        string randomVonNeumannAverageTimeFormatted = randomVonNeumannAverageTime.ToString(@"hh\:mm\:ss\.fff");

        string starTimeStndDevFormatted = starTimeStndDev.ToString(@"hh\:mm\:ss\.fff");
        string circleTimeStndDevFormatted = circleTimeStndDev.ToString(@"hh\:mm\:ss\.fff");
        string vonNeumannTimeStndDevFormatted = vonNeumannTimeStndDev.ToString(@"hh\:mm\:ss\.fff");
        string euclideanCircleTimeStndDevFormatted = euclideanCircleTimeStndDev.ToString(@"hh\:mm\:ss\.fff");
        string euclideanVonNeumannTimeStndDevFormatted = euclideanVonNeumannTimeStndDev.ToString(@"hh\:mm\:ss\.fff");
        string randomCircleTimeStndDevFormatted = randomCircleTimeStndDev.ToString(@"hh\:mm\:ss\.fff");
        string randomVonNeumannTimeStndDevFormatted = randomVonNeumannTimeStndDev.ToString(@"hh\:mm\:ss\.fff");

        //puts all the data into an excel sheet
        ExcelPackage.LicenseContext = OfficeOpenXml.LicenseContext.NonCommercial; //Set License Context for EPPlus

        var dataTable = new DataTable();

        // Define columns
        dataTable.Columns.AddRange(new[]
        {
            new DataColumn("Dataset", typeof(string)),
            new DataColumn("Number of Input Neurons", typeof(int)),
            new DataColumn("Number of Hidden Neurons", typeof(int)),
            new DataColumn("Number of Output Neurons", typeof(int)),
            new DataColumn("Number of Particles", typeof(int)),
            new DataColumn("Maximum Number of Iterations Per Run", typeof(int)),
            new DataColumn("Target Accuracy Per Run (%)", typeof(double)),
            new DataColumn("Frequency of Euclidean Topologies Changing", typeof(int)),
            new DataColumn("Number of Runs", typeof(int)),
            new DataColumn("Method", typeof(string)),
            new DataColumn("Average Accuracy", typeof(double)),
            new DataColumn("Standard Deviation", typeof(double)),
            new DataColumn("Average Runtime", typeof(string)),
            new DataColumn("Runtime Standard Deviation", typeof(string))
        });

        // Insert data
        dataTable.Rows.Add(dataset, numInput, numHidden, numOutput, numParticles, maxIterations, (1 - exitError) * 100, mod, runs, "Star", starAverageAccuracy, starStndDev, starAverageTimeFormatted, starTimeStndDevFormatted);
        dataTable.Rows.Add(dataset, numInput, numHidden, numOutput, numParticles, maxIterations, (1 - exitError) * 100, mod, runs, "Static Circle", circleAverageAccuracy, circleStndDev, circleAverageTimeFormatted, circleTimeStndDevFormatted);
        dataTable.Rows.Add(dataset, numInput, numHidden, numOutput, numParticles, maxIterations, (1 - exitError) * 100, mod, runs, "Static Von Neumann", vonNeumannAverageAccuracy, vonNeumannStndDev, vonNeumannAverageTimeFormatted, vonNeumannTimeStndDevFormatted);
        dataTable.Rows.Add(dataset, numInput, numHidden, numOutput, numParticles, maxIterations, (1 - exitError) * 100, mod, runs, "Euclidean Circle", euclideanCircleAverageAccuracy, euclideanCircleStndDev, euclideanCircleAverageTimeFormatted, euclideanCircleTimeStndDevFormatted);
        dataTable.Rows.Add(dataset, numInput, numHidden, numOutput, numParticles, maxIterations, (1 - exitError) * 100, mod, runs, "Euclidean Von Neumann", euclideanVonNeumannAverageAccuracy, euclideanVonNeumannStndDev, euclideanVonNeumannAverageTimeFormatted, euclideanVonNeumannTimeStndDevFormatted);
        dataTable.Rows.Add(dataset, numInput, numHidden, numOutput, numParticles, maxIterations, (1 - exitError) * 100, mod, runs, "Random Circle", randomCircleAverageAccuracy, randomCircleStndDev, randomCircleAverageTimeFormatted, randomCircleTimeStndDevFormatted);
        dataTable.Rows.Add(dataset, numInput, numHidden, numOutput, numParticles, maxIterations, (1 - exitError) * 100, mod, runs, "Random Von Neumann", randomVonNeumannAverageAccuracy, randomVonNeumannStndDev, randomVonNeumannAverageTimeFormatted, randomVonNeumannTimeStndDevFormatted);
        
        var filePath = @"C:\Users\andre\source\repos\NeuralNetworkPSOTraining\NeuralNetworkPSOTraining\Collected Data\" + dataset +  ".xlsx"; //Set the path to your file
        FileInfo file = new FileInfo(filePath);

        using (var package = new ExcelPackage(file))
        {
            var worksheetName = $"Sheet{package.Workbook.Worksheets.Count + 1}";
            var worksheet = package.Workbook.Worksheets.Add(worksheetName); //Creating worksheet
            worksheet.Cells["A1"].LoadFromDataTable(dataTable, true); //Load data to worksheet
            package.Save();
        }

        Console.WriteLine("Spreadsheet created successfully");
    }

    private static double[][] OneHotEncode(int[] labels, int numClasses)
    {
        // Initialize a 2D array for the one-hot encoded labels.
        double[][] oneHot = new double[labels.Length][];
        for (int i = 0; i < labels.Length; i++)
        {
            // Initialize the inner array.
            oneHot[i] = new double[numClasses];

            // Set the index corresponding to the label to 1.
            oneHot[i][labels[i]] = 1.0;
        }
        return oneHot;
    }

    private static double GetAverage(double[] values)
    {
        double average = 0;

        for (int i = 0; i < values.Length; i++)
        {
            average = average + values[i];
        }

        average = average / values.Length;

        return average;
    }

    private static double StandardDeviation(double[] values)
    {
        if(values.Length > 1)
        {
            double avg = GetAverage(values);
            double sum = 0.0;

            foreach (int val in values)
            {
                sum += Math.Pow((val - avg), 2.0);
            }

            double variance = sum / (double)(values.Length - 1);
            return Math.Sqrt(variance);
        }
        else 
        { 
            return 0; 
        }

    }
    private static double TimeSpanStandardDeviation(TimeSpan[] times)
    {
        if (times.Length > 1)
        {
            double[] doubleTimes = times.Select(time => (double)time.Ticks).ToArray();
            double avg = doubleTimes.Average();
            double sum = doubleTimes.Sum(d => (d - avg) * (d - avg));
            return Math.Sqrt(sum / doubleTimes.Length);
        }
        else
        {
            return 0;
        }
    }

    private static int GetInputNeurons(string dataset)
    {
        if (dataset == "Iris")
        {
            return 4;
        }
        else if (dataset == "car")
        {
            return 6;
        }
        else if (dataset == "phone")
        {
            return 20;
        }
        else if (dataset == "diabetes")
        {
            return 7;
        }
        else if(dataset == "mnist")
        {
            return 784;
        }
        else
        {
            throw new Exception("Not a valid Dataset");
        }
    }
    private static int GetOutputNeurons(string dataset)
    {
        if (dataset == "Iris")
        {
            return 3;
        }
        else if (dataset == "car")
        {
            return 4;
        }
        else if (dataset == "phone")
        {
            return 4;
        }
        else if (dataset == "diabetes")
        {
            return 2;
        }
        else if (dataset == "mnist")
        {
            return 10;
        }
        else
        {
            throw new Exception("Not a valid Dataset");
        }
    }
}