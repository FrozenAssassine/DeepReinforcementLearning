using System.Collections;
using System.Linq;
using TMPro;
using UnityEngine;
using static UnityEngine.GraphicsBuffer;

public class Agent3 : MonoBehaviour
{
    [SerializeField] float playerSpeed = 5;
    [SerializeField] float jumpHeight = 5;
    [SerializeField] float epsilon = 1f;
    [SerializeField] float epsilonMin = 0;
    [SerializeField] float gamma = 0.9f;
    [SerializeField] int numEpisodes = 1000;
    [SerializeField] GameObject Player;
    [SerializeField] GameObject[] Obstacles;
    [SerializeField] GameObject[] Boxes;
    [SerializeField] GameObject Goal;
    [SerializeField] GameObject Floor;
    [SerializeField] TMP_Text overviewDisplay;
    [SerializeField] TMP_Text epsilonDisplay;

    bool hitBox = false;
    bool hitGoal = false;
    bool hitObstacle = false;
    bool isGrounded = false;
    int trainedEpochs = 0;
    NNModel model;
    bool done = false;
    Vector3 initialPlayerPosition;
    Quaternion initialPlayerRotation;
    int passedCount = 0;
    int failedCount = 0;
    float passedFailedRatio = 0.0f;
    float totalTime = 0;
    int totalStates => Boxes.Length + Obstacles.Length + 3;
    int jumpCount = 0;

    void Start()
    {
        model = NetworkBuilder.Create()
            .Stack(new InputLayer(totalStates)) //obst1 distance, obst2 distance, rotationY
            .Stack(new DenseLayer(100, ActivationType.Relu))
            .Stack(new OutputLayer(2, ActivationType.Softmax))
            .Build(false);

        initialPlayerPosition = Player.transform.position;
        initialPlayerRotation = Player.transform.rotation;

        StartCoroutine(TrainModel());
    }


    public void LoadWeights()
    {
        model.Load("C:\\Users\\juliu\\desktop\\weights.cool");
    }
    public void SaveWeights()
    {
        model.Save("C:\\Users\\juliu\\desktop\\weights.cool");
    }

    float oldTimeScale = 0;
    public void PauseResume()
    {
        if (Time.timeScale == 0)
            Time.timeScale = oldTimeScale;
        else
        {
            oldTimeScale = Time.timeScale;
            Time.timeScale = 0;
        }
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
        jumpCount = 0;
    }

    void WalkForward()
    {
        Player.transform.Translate(Vector3.left * playerSpeed * Time.deltaTime);
    }
    void JumpPlayer()
    {
        if (isGrounded)
        {
            jumpCount++;
            Player.GetComponent<Rigidbody>().AddForce(Vector3.up * jumpHeight, ForceMode.Impulse);
            isGrounded = false;
        }
    }

    float[] GetStateArray()
    {
        float[] state = new float[totalStates];

        for (int i = 0; i < Boxes.Length; i++)
            state[i] = Player.transform.position.x - Boxes[i].transform.position.x;

        for (int i = 0; i < Obstacles.Length; i++)
            state[Boxes.Length + i] = Player.transform.position.x - Obstacles[i].transform.position.x;
        
        state[totalStates - 1] = Player.transform.position.y - initialPlayerPosition.y;
        state[totalStates - 2] = Player.GetComponent<Rigidbody>().velocity.x;
        state[totalStates - 3] = isGrounded ? 1.0f : 0.0f;
        return state;
    }

    void PerformAction(int action)
    {
        switch (action)
        {
            case 0:
                break;
            case 1:
                JumpPlayer();
                break;
        }
    }
    float CalculateReward()
    {
        float reward = -0.001f;

        if (hitBox)
        {
            reward += 0.005f;
            hitBox = false;
        }

        if (hitGoal)
        {
            float timeFactor = Mathf.Clamp(1.0f - (Time.time - totalTime) / 10, 0.2f, 1.0f);
            reward += 1.0f * timeFactor;
            passedCount++;
            hitGoal = false;
        }
        else if (hitObstacle)
        {
            reward = -1.0f;
            failedCount++;
            hitObstacle = false;
        }
        return reward;
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

                if (reward != -0.001f && hitBox == false)
                {
                    if (jumpCount < 3 || jumpCount > 3)
                        reward -= 0.03f * jumpCount;
                    else if (jumpCount == 3)
                        reward += 0.04f;

                    //jumping on non ground level
                    if(Player.transform.position.y - initialPlayerPosition.y > 0.5)
                    {
                        reward += 0.04f;
                    }

                    //update the training values:
                    float[] qValues = model.FeedForward(state).ToArray();
                    float qTarget = reward + gamma * ArgsMax(model.FeedForward(state));
                    qValues[action] = qTarget;
                    model.Train(state, qValues, 0.05f);

                    UpdateUI(reward, action, qTarget);

                    epsilon = Mathf.Max(epsilon * 0.9f, epsilonMin);
                }

                yield return null;
            }
        }
    }

    void UpdateUI(float reward, int action, float qTarget)
    {
        epsilonDisplay.text = epsilon.ToString();

        overviewDisplay.text =
            $"Epochs: {trainedEpochs++}\n" +
            $"Reward: {reward}\n" +
            $"Passed: {passedCount}\n" +
            $"Failed: {failedCount}\n" +
            $"Target: {qTarget}\n" +
            $"Action: {action}\n" +
            $"P/F: {passedFailedRatio}";
    }


    void OnCollisionEnter(Collision collision)
    {
        //detect collision with goal or obstacle
        if (collision.gameObject == Goal)
        {
            hitGoal = true;
            done = true;
        }

        if (collision.gameObject.tag.Equals("Obstacle"))
        {
            hitObstacle = true;
            done = true;
        }

        //detect collision with ground for jumping
        if (collision.gameObject.layer == 3) //ground
        {
            isGrounded = true;
            if (collision.gameObject.tag.Equals("CanJumpOn"))
            {
                hitBox = true;
            }
        }
    }

    void Update()
    {
        WalkForward();
        if (done)
        {
            done = false;
            ResetPlayer();
        }
    }
}
