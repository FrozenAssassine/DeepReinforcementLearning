using System;
using System.Diagnostics;
using System.IO;
using System.Linq;

public class NNModel
{
    private NeuralNetwork nn;
    private bool useCuda = false;
    private BaseLayer[] layers;
    public NNModel(BaseLayer[] layers, bool useCuda = true)
    {
        this.layers = layers;

        if (layers.Length < 3)
        {
            throw new Exception("You need at least one input, hidden and output layer");
        }

        nn = new NeuralNetwork(layers);
    }

    public float[] FeedForward(float[] inputs)
    {
        return nn.FeedForward_CPU(inputs);
    }

    public float[] Predict(float[] input, bool output = false)
    {
        float[] prediction = null;
        var time = BenchmarkExtension.Benchmark(() =>
        {
            prediction = nn.FeedForward_CPU(input);
        });
        if (output)
            Console.WriteLine("Prediction time " + time);
        return prediction;
    }


    public void Train(float[] inputs, float[] desired, float learningRate)
    {
        nn.Train_CPU(inputs, desired,learningRate);
    }

    public void Save(string path)
    {
        Console.WriteLine("Saving model data to file");
        var ms = new MemoryStream();
        nn.Save(ms);
        File.WriteAllBytes(path, ms.ToArray());
        Console.WriteLine($"Saved to {path}");
    }

    public void Load(string path)
    {
        Console.WriteLine("Loading model data from file");
        var bytes = File.ReadAllBytes(path);
        var ms = new MemoryStream(bytes);
        nn.Load(ms);
        Console.WriteLine($"Loaded from {path}");
    }

    //only use cuda evaluation while training, because gpu memory gets freed after training
    public (float percent, int count, int correct) Evaluate(float[][] x, float[][] y, bool predictOnCuda = true, bool output = true)
    {
        int correct = 0;

        if (predictOnCuda && this.useCuda)
        {
            for (int i = 0; i < x.Length; i++)
            {
                float[] prediction = new float[this.layers[^1].Size];
                //CudaAccel.Predict(x[i], prediction);
                if (MathHelper.GetMaximumIndex(y[i]) == MathHelper.GetMaximumIndex(prediction))
                    correct++;
            }
        }
        else
        {
            for (int i = 0; i < x.Length; i++)
            {
                if (MathHelper.GetMaximumIndex(y[i]) == MathHelper.GetMaximumIndex(nn.FeedForward_CPU(x[i])))
                    correct++;
            }
        }

        float accuracy = (float)correct / x.Length;
        if(output)
            Console.WriteLine($"Evaluation: {x.Length}/{correct} ({accuracy.ToString().Replace(",", ".")}) ({(int)(accuracy * 100.0f)}%)");

        return (accuracy, x.Length, correct);
    }
    
    public void Summary()
    {
        nn.Summary();
    }
}
