using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System.Linq;

// Robot controller that will be trained with neural network
public class RobotController : MonoBehaviour
{
    [Header("Neural Network")]
    public int inputNeurons = 5;
    public int hiddenNeurons = 20;
    public int outputNeurons = 2;
    private NNModel nnmodel;

    private List<float[]> trainingInputs = new List<float[]>();
    private List<float[]> trainingOutputs = new List<float[]>();
    private float performanceScore = 0;

    [Header("Robot Movement")]
    public float maxSpeed = 2.0f;
    public float maxRotationSpeed = 120.0f;
    public float motorLeftOutput = 0f;
    public float motorRightOutput = 0f;

    [Header("Sensor")]
    public float maxSensorDistance = 5.0f;
    public Transform sensorPivot;
    
    public int scanPoints = 5;
    private float[] sensorReadings;

    [Header("Training")]
    public float rewardMultiplier = 1.0f;
    public float collisionPenalty = -10.0f;
    private float timeSinceLastCollision = 0f;
    private float totalReward = 0f;
    private int episodeCount = 0;
    private bool isTraining = true;

    // Episode management
    public float episodeLength = 30f;
    private float episodeTimer = 0f;
    public float trainingInterval = 100f; // Train every 100 episodes
    private float trainingTimer = 0f;
    public int batchSize = 1000; // How many samples to collect before training
    private int samplesCollected = 0;

    private bool isTrainingCoroutineRunning = false;

    void Start()
    {
        episodeLength = Random.Range(5, 100);

        // Initialize neural network using the provided library
        InitializeNeuralNetwork();

        // Initialize sensor readings array
        sensorReadings = new float[scanPoints];
    }

    void InitializeNeuralNetwork()
    {
        // Create neural network using the provided NetworkBuilder
        nnmodel = NetworkBuilder.Create()
            .Stack(new InputLayer(inputNeurons))
            .Stack(new DenseLayer(hiddenNeurons, ActivationType.Relu))
            .Stack(new DenseLayer(hiddenNeurons, ActivationType.Relu))
            .Stack(new OutputLayer(outputNeurons, ActivationType.Softmax))
            .Build(true); // Set to true to use GPU acceleration if available

        Debug.Log("Neural Network initialized");
        nnmodel.Summary();
    }

    void Update()
    {
        if (isTraining)
        {
            // Update timers
            episodeTimer += Time.deltaTime;
            trainingTimer += Time.deltaTime;
            timeSinceLastCollision += Time.deltaTime;

            // Perform sensor scan
            ScanEnvironment();

            // Feed sensor data to neural network for prediction
            float[] inputs = sensorReadings;
            float[] outputs = nnmodel.FeedForward(inputs);

            // Apply outputs to robot motors (scale from 0-1 to -1 to 1)
            motorLeftOutput = (outputs[0] * 2f) - 1f;
            motorRightOutput = (outputs[1] * 2f) - 1f;

            // Move the robot
            MoveRobot(motorLeftOutput, motorRightOutput);

            // Calculate reward based on time survived and forward motion
            float movementReward = CalculateMovementReward();
            float reward = (Time.deltaTime * rewardMultiplier) + movementReward;
            totalReward += reward;
            performanceScore += reward;

            // Collect training data with current state and action
            CollectTrainingData(inputs, new float[] { (motorLeftOutput + 1f) / 2f, (motorRightOutput + 1f) / 2f }, reward);

            // Check if episode is over
            if (episodeTimer >= episodeLength)
            {
                float episodePerformance = performanceScore / episodeLength;
                Debug.Log($"Episode {episodeCount} completed. Total reward: {totalReward}, Performance: {episodePerformance}");
                episodeCount++;
                performanceScore = 0;
                ResetEpisode();
            }

            // Train the network periodically without blocking
            if (trainingTimer >= trainingInterval && trainingInputs.Count >= batchSize && !isTrainingCoroutineRunning)
            {
                StartCoroutine(TrainNetworkWithCollectedData());
                trainingTimer = 0f;
            }
        }
    }


    float CalculateMovementReward()
    {
        float distanceReward = sensorReadings[2];
        float speedReward = (motorLeftOutput + motorRightOutput) * 0.5f;
        float turnPenalty = -Mathf.Abs(motorLeftOutput - motorRightOutput) * 0.05f;
        return distanceReward + speedReward + turnPenalty;
    }

    void CollectTrainingData(float[] inputs, float[] outputs, float reward)
    {
        trainingInputs.Add(inputs.Clone() as float[]);

        float[] desiredOutputs = new float[outputNeurons];

        if (reward >= 0)
        {
            for (int i = 0; i < outputNeurons; i++)
            {
                desiredOutputs[i] = outputs[i];
            }
        }
        else
        {
            if (FindMostBlockedDirection() < scanPoints / 2)
            {
                desiredOutputs[0] = 0.4f; // Slow left motor
                desiredOutputs[1] = 0.6f; // Fast right motor
            }
            else
            {
                desiredOutputs[0] = 0.6f; // Fast left motor
                desiredOutputs[1] = 0.4f; // Slow right motor
            }
        }

        trainingOutputs.Add(desiredOutputs);
        samplesCollected++;

        // Limit the size of the training data to conserve memory
        if (trainingInputs.Count > batchSize * 2)
        {
            trainingInputs.RemoveAt(0);
            trainingOutputs.RemoveAt(0);
        }
    }

    int FindMostBlockedDirection()
    {
        // Find the index of the shortest sensor reading
        float minDistance = float.MaxValue;
        int minIndex = 0;

        for (int i = 0; i < sensorReadings.Length; i++)
        {
            if (sensorReadings[i] < minDistance)
            {
                minDistance = sensorReadings[i];
                minIndex = i;
            }
        }
        return minIndex;
    }

    void TrainNetworkWithCollectedData()
    {
        if (trainingInputs.Count < 10)
        {
            Debug.Log("Not enough training samples collected yet");
            return;
        }

        Debug.Log($"Training network with {trainingInputs.Count} samples...");

        // Convert lists to arrays for training
        float[][] inputsArray = trainingInputs.ToArray();
        float[][] outputsArray = trainingOutputs.ToArray();

        nnmodel.Train(inputsArray, outputsArray, 50, 0.003f);

        Debug.Log("Network training completed");

        // Save the model after training
        SaveWeights();

        // Clear some of the older training data
        int samplesToKeep = batchSize / 2;
        if (trainingInputs.Count > samplesToKeep)
        {
            trainingInputs = trainingInputs.Skip(trainingInputs.Count - samplesToKeep).ToList();
            trainingOutputs = trainingOutputs.Skip(trainingOutputs.Count - samplesToKeep).ToList();
        }
        isTrainingCoroutineRunning = false;
    }

    void ScanEnvironment()
    {
        float[] angles = { -30, -15, 0, 15, 30 };

        for (int i = 0; i < angles.Length; i++)
        {
            Vector3 dir = Quaternion.Euler(0, angles[i], 0) * transform.forward;
            Ray ray = new Ray(sensorPivot.position, dir);
            RaycastHit hit;

            if (Physics.Raycast(ray, out hit, maxSensorDistance))
            {
                sensorReadings[i] = hit.distance / maxSensorDistance;
                Debug.DrawRay(sensorPivot.position, dir * hit.distance, Color.red);
            }
            else
            {
                sensorReadings[i] = 1.0f;
                Debug.DrawRay(sensorPivot.position, dir * maxSensorDistance, Color.green);
            }
        }
    }

    void MoveRobot(float leftMotor, float rightMotor)
    {
        float forwardSpeed = (leftMotor + rightMotor) * 0.5f * maxSpeed;
        float rotationSpeed = (rightMotor - leftMotor) * maxRotationSpeed;

        transform.Translate(Vector3.forward * forwardSpeed * Time.deltaTime);
        transform.Rotate(Vector3.up, rotationSpeed * Time.deltaTime);
    }

    void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Wall"))
        {
            totalReward += collisionPenalty;
            performanceScore += collisionPenalty;
            timeSinceLastCollision = 0f;

            //collect negative training data for this collision
            float[] inputs = sensorReadings.Clone() as float[];
            float[] currentOutputs = new float[] { (motorLeftOutput + 1f) / 2f, (motorRightOutput + 1f) / 2f };
            CollectTrainingData(inputs, currentOutputs, collisionPenalty);

            ResetEpisode();
        }
    }

    void ResetEpisode()
    {
        Vector3 startPosition = new Vector3(Random.Range(-10, 10), 5f, Random.Range(-10, 10));
        transform.position = startPosition;
        transform.rotation = Quaternion.Euler(0, Random.Range(-30, 30), 0);

        episodeTimer = 0f;
        timeSinceLastCollision = 0f;
        totalReward = 0f;
    }

    void SaveWeights()
    {
        //nnmodel.Save(Application.dataPath + "D:\\robot\\robot_model.cool");
    }
}