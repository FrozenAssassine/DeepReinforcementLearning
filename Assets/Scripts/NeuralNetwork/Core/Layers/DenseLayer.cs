﻿using System.IO;
using System.Threading.Tasks;
using UnityEngine;

public class DenseLayer : BaseLayer
{
    public DenseLayer(int size, ActivationType activation)
    {
        this.Size = size;
        this.ActivationFunction = activation;
    }

    public override void Train(float[] desired, float learningRate)
    {
        Parallel.For(0, this.Size, (idx) =>
        {
            float err = 0.0f;
            int index = idx * this.PreviousLayer.Size;

            for (int j = 0; j < this.NextLayer.Size; j++)
            {
                err += (this.NextLayer.Errors[j] * this.NextLayer.Weights[j * this.Size + idx]);
            }
            float error = err * ActivationFunctions.ActivationDeriv(this.NeuronValues[idx], this.ActivationFunction);
            this.Errors[idx] = error;

            error *= learningRate;

            for (int j = 0; j < this.PreviousLayer.Size; j++)
            {
                this.Weights[index + j] += error * this.PreviousLayer.NeuronValues[j];
            }
            this.Biases[idx] += error;
        });
    }

    public override void FeedForward()
    {
        Parallel.For(0, this.Size, (idx) => {
            float sum = 0.0f;
            int index = idx * this.PreviousLayer.Size;
            for (int j = 0; j < this.PreviousLayer.Size; j++)
            {
                sum += this.PreviousLayer.NeuronValues[j] * this.Weights[index + j];
            }
            this.NeuronValues[idx] = ActivationFunctions.Activation(sum + this.Biases[idx], this.ActivationFunction);
        });
    }

    public override void Load(BinaryReader br)
    {
        LayerSaveLoadFunction.Load(this, br);
    }

    public override void Save(BinaryWriter bw)
    {
        LayerSaveLoadFunction.Save(this, bw);
    }

    public override void Summary()
    {
        Debug.Log($"Dense Layer of {Size} Neurons and {Weights.Length} Weights");
    }

    public override void Initialize(int inputCount, int outputCount)
    {
        LayerInitialisationHelper.InitializeLayer(this, inputCount, outputCount);
    }

    public override void InitializeCuda(int index)
    {
        //CudaAccel.InitDenseLayer(index, this.PreviousLayer.Size, this.Size, this.Biases, this.Weights, this.NeuronValues, this.Errors, this.ActivationFunction);
    }
}