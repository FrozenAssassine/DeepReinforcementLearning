﻿using System.Runtime.InteropServices;

internal static class CudaAccel
{
    const string DDL_PATH = "F:\\C#\\NNFromScratch\\x64\\Release\\CudaWrapper.dll"; //your cuda dll path

    [DllImport(DDL_PATH, CallingConvention = CallingConvention.Cdecl)]
    public static extern void Train(float[] inputs, float[] desired, int size, float learningRate);

    [DllImport(DDL_PATH, CallingConvention = CallingConvention.Cdecl)]
    public static extern void Init(int totalLayers);

    [DllImport(DDL_PATH, CallingConvention = CallingConvention.Cdecl)]
    public static extern void DoneTraining();
    [DllImport(DDL_PATH, CallingConvention = CallingConvention.Cdecl)]
    [return: MarshalAs(UnmanagedType.I1)]
    public static extern bool CheckCuda();

    [DllImport(DDL_PATH, CallingConvention = CallingConvention.Cdecl)]
    public static extern void Predict(float[] data, float[] prediction);


    [DllImport(DDL_PATH, CallingConvention = CallingConvention.Cdecl)]
    public static extern void InitInputLayer(
        int layerIndex,
        int size,
        float[] biases,
        float[] weights,
        float[] neuronValues,
        float[] errors,
        ActivationType activation);

    [DllImport(DDL_PATH, CallingConvention = CallingConvention.Cdecl)]
    public static extern void InitOutputLayer(
        int layerIndex,
        int prevSize,
        int size,
        float[] biases,
        float[] weights,
        float[] neuronValues,
        float[] errors,
        ActivationType activation);

    [DllImport(DDL_PATH, CallingConvention = CallingConvention.Cdecl)]
    public static extern void InitDenseLayer(
        int layerIndex,
        int prevSize,
        int size,
        float[] biases,
        float[] weights,
        float[] neuronValues,
        float[] errors,
        ActivationType activation);

    [DllImport(DDL_PATH, CallingConvention = CallingConvention.Cdecl)]
    public static extern void InitLSTMLayer(
        int layerIndex,
        int prevSize,
        int size,
        float[] biases,
        float[] neuronValues,
        float[] errors,
        float[] WeightsInput,
        float[] WeightsForget,
        float[] WeightsOutput,
        float[] WeightsCandidate,
        float[] CellState,
        float[] OutputGate,
        float[] ForgetGate,
        float[] InputGate,
        float[] CandidateCellState,
        float[] inputGateGradients,
        float[] forgetGateGradients,
        float[] outputGradients,
        float[] candidateCellGradients
    );
}