using System.Collections;
using System.Linq;
using TMPro;
using UnityEngine;

public class Agent2 : MonoBehaviour
{
    [SerializeField] float playerSpeed = 5;
    [SerializeField] float jumpHeight = 5;
    [SerializeField] float epsilonReductionRate = 0.001f;
    [SerializeField] float epsilon = 1f;
    [SerializeField] float epsilonMin = 0.001f;
    [SerializeField] float gamma = 0.9f;
    [SerializeField] int numEpisodes = 1000;
    [SerializeField] GameObject Player;
    [SerializeField] GameObject Box1;
    [SerializeField] GameObject Obstacle2;
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
    int jumpCount = 0;

    void Start()
    {
        model = NetworkBuilder.Create()
            .Stack(new InputLayer(4))
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
        //jump only if the player is grounded and not in cooldown
        if (isGrounded)
        {
            jumpCount++;
            Player.GetComponent<Rigidbody>().AddForce(Vector3.up * jumpHeight, ForceMode.Impulse);
            isGrounded = false;
        }
    }

    IEnumerator TrainModel()
    {
        for (int episode = 0; episode < numEpisodes; episode++)
        {
            bool isEpisodeDone = false;
            bool done = false;

            while (!isEpisodeDone)
            {
                float reward = -0.005f;
                int action = 0;
                float boxDist = Player.transform.position.x - Box1.transform.position.x;
                float obst2Dist = Player.transform.position.x - Obstacle2.transform.position.x;
                float rotationY = Player.transform.rotation.y;
                float height = Player.transform.position.y - initialPlayerPosition.y;
                float[] state = { boxDist, obst2Dist, height, Time.time - totalTime };

                if (Random.value < epsilon)
                    action = Random.Range(0, 2);
                else
                    action = ArgsMaxIndex(model.Predict(state));

                if(action == 1)
                    JumpPlayer();

                if (hitBox)
                {
                    reward = 0.05f;
                    hitBox = false;
                }

                //check for goal and obstacle conditions
                if (hitGoal)
                {
                    //lower reward if it took longer:
                    float timeFactor = Mathf.Clamp(1.0f - (Time.time - totalTime) / 20, 0.1f, 0.8f);
                    reward += 0.8f * timeFactor;

                    if (jumpCount == 1)
                        reward += 0.1f;
                    else
                    {
                        reward -= 0.3f;
                    }

                    passedCount++;
                    hitGoal = false;
                }
                else if (hitObstacle)
                {
                    reward = -1f;
                    failedCount++;
                    hitObstacle = false;
                }

                //retrain the model when the reward is not 0
                if (reward != -0.005f)
                {
                    float maxQValueNext = ArgsMax(model.FeedForward(state));
                    float qTarget = reward + gamma * maxQValueNext;

                    float[] qValues = model.FeedForward(state);
                    qValues[action] = qTarget;

                    model.Train(state, qValues, 0.05f);
      
                    epsilon = Mathf.Max(epsilon * 0.99f, epsilonMin);

                    epsilonDisplay.text = $"Epsilon: {epsilon}";
                    overviewDisplay.text = $"Epochs: {trainedEpochs++}\nReward: {reward}\nPassed: {passedCount}\nFailed: {failedCount}\nTarget: {qTarget}\nQNext: {maxQValueNext}\nAction: {action}\nP/F: {passedFailedRatio}";

                    totalTime = Time.time;

                    if (done)
                    {
                        isEpisodeDone = true;
                        done = false;
                    }
                }

                yield return null;
            }

            ResetPlayer();
        }
    }

    void OnCollisionEnter(Collision collision)
    {
        //detect collision with goal or obstacle
        if (collision.gameObject == Goal)
        {
            hitGoal = true;
            done = true;
        }
        else if (collision.gameObject == Obstacle2)
        {
            hitObstacle = true;
            done = true;
        }
        else if(collision.gameObject == Box1)
        {
            hitBox = true;
        }

        //detect collision with ground for jumping
        if (collision.gameObject.CompareTag("Ground"))
        {
            isGrounded = true;
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
