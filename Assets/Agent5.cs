using System.Collections;
using System.Collections.Generic;
using System.Linq;
using TMPro;
using UnityEngine;

public class Agent5 : MonoBehaviour
{
    [SerializeField] float playerSpeed = 5;
    [SerializeField] float jumpHeight = 5;
    [SerializeField] float epsilon = 1f;
    [SerializeField] float epsilonMin = 0;
    [SerializeField] float gamma = 0.9f;
    [SerializeField] int numEpisodes = 1000;
    [SerializeField] GameObject Player;

    int trainedEpochs = 0;
    NNModel model;
    bool done = false;
    Vector3 initialPlayerPosition;
    Quaternion initialPlayerRotation;
    int passedCount = 0;
    int failedCount = 0;
    float passedFailedRatio = 0.0f;
    float totalTime = 0;
    BirdController controller;
    bool hitObstacle = false;

    void Start()
    {
        model = NetworkBuilder.Create()
            .Stack(new InputLayer(3))
            .Stack(new DenseLayer(40, ActivationType.Relu))
            .Stack(new OutputLayer(2, ActivationType.Softmax))
            .Build(false);

        initialPlayerPosition = Player.transform.position;
        initialPlayerRotation = Player.transform.rotation;

        controller = Player.GetComponent<BirdController>();

        StartCoroutine(TrainModel());
    }

    int ArgsMaxIndex(float[] items)
    {
        return System.Array.IndexOf(items, items.Max());
    }
    float ArgsMax(float[] items)
    {
        return items.Max();
    }

    void ResetPlayer()
    {
        Player.transform.position = initialPlayerPosition;
        Player.transform.rotation = initialPlayerRotation;
        Player.GetComponent<Rigidbody>().velocity = Vector3.zero;
    }


    float[] GetStateArray()
    {
        float[] state = new float[3];

        state[0] = controller.GetVelocity();
        state[1] = controller.GetDistanceToTopObstacle();
        state[2] = controller.GetDistanceToBottomObstacle();
        return state;
    }

    void PerformAction(int action)
    {
        switch (action)
        {
            case 0:
                break;
            case 1:
                controller.Flap();
                break;
        }
    }
    float CalculateReward()
    {
        float reward = -0.001f;


    }

    IEnumerator TrainModel()
    {
        for (int episode = 0; episode < numEpisodes; episode++)
        {
            ResetPlayer();
            bool isEpisodeDone = false;

            while (!isEpisodeDone)
            {
                int action;
                float[] state = GetStateArray();

                // Epsilon-greedy action selection
                if (Random.value < epsilon)
                    action = Random.Range(0, 2);
                else
                    action = ArgsMaxIndex(model.Predict(state));

                PerformAction(action);

                float reward = CalculateReward();
                isEpisodeDone = hitGoal || hitObstacle;

                if (reward != -0.001f)
                {
                    //jumping on non ground level
                    if (Player.transform.position.y - initialPlayerPosition.y > 0.5)
                    {
                        reward += 0.04f;
                    }

                    //update the training values:
                    float[] qValues = model.FeedForward(state).ToArray();
                    float qTarget = reward + gamma * ArgsMax(model.FeedForward(state));
                    qValues[action] = qTarget;
                    model.Train(state, qValues, 0.05f);

                    epsilon = Mathf.Max(epsilon * 0.9f, epsilonMin);
                }

                yield return null;
            }
        }
    }

    private void OnCollisionEnter(Collision collision)
    {
        if(collision.gameObject.layer == 6)
        {
            hitObstacle = true;
        }
    }


    void Update()
    {
        if (done)
        {
            done = false;
            ResetPlayer();
        }
    }
}
