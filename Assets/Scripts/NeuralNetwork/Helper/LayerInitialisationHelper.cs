using JetBrains.Annotations;
using System;
using System.Numerics;

internal class LayerInitialisationHelper
{
    public static float[] XavierInitializeWeights(int fanIn, int fanOut, int totalWeights)
    {
        float limit = (float)Math.Sqrt(6.0 / (fanIn + fanOut));
        float[] weights = new float[totalWeights];

        for (int i = 0; i < totalWeights; i++)
        {
            weights[i] = MathHelper.RandomFloat(-limit, limit);
        }
        return weights;
    }

    public static void FillRandom(float[] biases)
    {
        for (int i = 0; i < biases.Length; i++)
        {
            biases[i] = MathHelper.RandomFloat1_1();
        }
    }
    
    public static void InitializeLayer(BaseLayer layer, int inputCount, int outpuCount)
    {
        layer.Biases = new float[layer.Size];
        layer.NeuronValues = new float[layer.Size];
        layer.Errors = new float[layer.Size];

        //first layer does not have previousLayer:
        if(layer.PreviousLayer != null)
            layer.Weights = XavierInitializeWeights(inputCount, outpuCount, layer.Size * layer.PreviousLayer.Size);

        FillRandom(layer.Biases);
    }
}
