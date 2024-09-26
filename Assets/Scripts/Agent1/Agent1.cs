using System.Collections;
using System.Linq;
using System.Runtime.CompilerServices;
using TMPro;
using UnityEngine;

enum RunningState
{
    Train, Test, Pause
}

public class Agent1 : MonoBehaviour
{
    [SerializeField]float playerSpeed = 5;
    [SerializeField] float jumpHeight = 6;
    [SerializeField] float epsilonReductionRate = 0.001f;
    [SerializeField] float epsilon = 1f;
    [SerializeField] float epsilonMin = 0.01f;
    [SerializeField] float gamma = 0.9f;
    [SerializeField] int numEpisodes = 1000;

    [SerializeField] GameObject Player;
    [SerializeField] GameObject Obstacle;
    [SerializeField] GameObject Goal;
    [SerializeField] GameObject Checkpoint;
    [SerializeField] GameObject Floor;
    [SerializeField] TMP_Text overviewDisplay;
    [SerializeField] TMP_Text epsilonDisplay;

    RunningState runningState = RunningState.Train;
    bool hitGoal = false;
    bool hitObstacle = false;
    bool hitCheckPoint = false;
    int trainedEpochs = 0;
    NNModel model;
    bool done = false;
    Vector3 initialPlayerPosition;
    int passedCount = 0;
    int failedCount = 0;
    int jumpCount = 0;
    float passedFailedRatio = 0.0f;

    void Start()
    {
        Time.timeScale = 5;
        model = NetworkBuilder.Create()
            .Stack(new InputLayer(4))
            .Stack(new DenseLayer(12, ActivationType.Sigmoid))
            .Stack(new OutputLayer(2, ActivationType.Sigmoid))
            .Build(false);

        initialPlayerPosition = Player.transform.position;

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
    void ReduceEpsilon()
    {
        ////drastically reduce:
        //if(passedFailedRatio < 0 && trainedEpochs > 5)
        //    epsilon += (passedFailedRatio / 5);

        if (epsilon > epsilonMin)
        {
            epsilon = Mathf.Clamp(epsilon - epsilonReductionRate * trainedEpochs, epsilonMin, 1);
            epsilonDisplay.text = $"{runningState}\nEpsilon: {epsilon}";
        }
    }
    void ResetPlayer(Vector3 homePosition)
    {
        ReduceEpsilon();
        Player.transform.position = homePosition;
        Player.GetComponent<Rigidbody>().velocity = Vector3.zero;
    }
    void MovePlayer()
    {
        Player.transform.Translate(Vector3.left * playerSpeed * Time.deltaTime);
    }

    IEnumerator TrainModel()
    {
        for (int episode = 0; episode < numEpisodes; episode++)
        {
            bool isEpisodeDone = false;
            bool done = false;

            while (!isEpisodeDone)
            {
                float distanceToObstacle = Player.transform.position.x - Obstacle.transform.position.x;
                float height = Player.transform.position.y - Floor.transform.position.y;
                float[] state = { distanceToObstacle / 10.0f, height / (jumpHeight / 2.0f), (Time.time - lastJumpTime) / 2.0f, jumpCount == 0 ? 0.0f : jumpCount / 4.0f };
                float reward = 0;
                passedFailedRatio = failedCount - passedCount;

                //predict or generate random action
                int action;
                if (runningState == RunningState.Train)
                {
                    if (Random.value < epsilon)
                    {
                        action = Random.Range(0, 2);
                        Debug.Log("RANDOM NUMBER");
                    }
                    else
                    {
                        var pred = model.FeedForward(state);
                        action = ArgsMaxIndex(pred);
                    }
                }
                else // test state:
                {
                    var pred = model.FeedForward(state);
                    action = ArgsMaxIndex(pred);
                }

                //execute the action:
                if (action == 1) // Jump
                    JumpPlayer();

                //check for goal and obstacle conditions
                if (hitGoal)
                {
                    reward = (1 / state[3]) / 20; //more jumps lower reward
                    passedCount++;
                    hitGoal = false;
                }
                else if (hitObstacle)
                {
                    reward = jumpCount == 0 ? -1f : -0.5f;
                    failedCount++;
                    hitObstacle = false;
                }

                //retrain the model when the reward is not 0
                if (reward != 0)
                {
                    Debug.Log("Train with: Height: " + state[1] + " Distance: " + state[0] + "Lastjump: " + state[2] + "Jumpcount: " + state[3]);

                    float maxQValueNext = ArgsMax(model.FeedForward(state));
                    float qTarget = reward + gamma * maxQValueNext;

                    float[] qValues = model.FeedForward(state);
                    qValues[action] = qTarget;

                    if (runningState == RunningState.Train)
                        model.Train(state, qValues, 0.05f);

                    overviewDisplay.text = $"Epochs: {trainedEpochs++}\nReward: {reward}\nPassed: {passedCount}\nFailed: {failedCount}\nTarget: {qTarget}\nQNext: {maxQValueNext}\nAction: {action}\nP/F: {passedFailedRatio}";
                    //if(trainedEpochs > 4)
                    //{
                    //    runningState = RunningState.Test;
                    //}
                    
                    //reset variables:
                    jumpCount = 0;
                    state[3] = 0;
                    lastJumpTime = Time.time; //only log the time in the current session:

                    if (done)
                    {
                        isEpisodeDone = true;
                        done = false;
                    }
                }

                yield return null;
            }

            ResetPlayer(initialPlayerPosition); 
        }
    }

    bool isGrounded;
    float jumpCooldown = 1.0f;
    float lastJumpTime;
    void JumpPlayer()
    {
        //jump only if the player is grounded and not in cooldown
        if (isGrounded && Time.time - lastJumpTime > jumpCooldown)
        {
            jumpCount++;
            Player.GetComponent<Rigidbody>().AddForce(Vector3.up * jumpHeight, ForceMode.Impulse);
            isGrounded = false;
            lastJumpTime = Time.time;
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
        else if (collision.gameObject == Obstacle)
        {
            hitObstacle = true;
            done = true;
        }

        //detect collision with ground for jumping
        if (collision.gameObject.CompareTag("Ground"))
        {
            isGrounded = true;
        }
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other == Checkpoint)
        {
            hitCheckPoint = true;
        }
    }

    void Update()
    {
        MovePlayer();
        if (done)
        {
            done = false;
            ResetPlayer(initialPlayerPosition);
        }
    }
}
