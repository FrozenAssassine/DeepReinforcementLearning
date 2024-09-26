using NUnit.Framework;
using System.Collections;
using System.Linq;
using TMPro;
using UnityEngine;


public class Agent2 : MonoBehaviour
{
    [SerializeField] float playerSpeed = 5;
    [SerializeField] float rotateSpeed = 50;
    [SerializeField] float jumpHeight = 5;
    [SerializeField] float epsilonReductionRate = 0.001f;
    [SerializeField] float epsilon = 1f;
    [SerializeField] float epsilonMin = 0.01f;
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

    void Start()
    {
        //Time.timeScale = 5;
        model = NetworkBuilder.Create()
            .Stack(new InputLayer(4)) //obst1 distance, obst2 distance, rotationY
            .Stack(new DenseLayer(50, ActivationType.Sigmoid))
            .Stack(new OutputLayer(2, ActivationType.Sigmoid))
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
    }

    void WalkForward()
    {
        Player.transform.Translate(Vector3.left * playerSpeed * Time.deltaTime);
    }
    void RotateLeft()
    {
        Player.transform.Rotate(new Vector3(0,1,0) * rotateSpeed * Time.deltaTime);
    }
    void RotateRight()
    {
        Player.transform.Rotate(new Vector3(0, -1, 0) * rotateSpeed * Time.deltaTime);
    }

    void JumpPlayer()
    {
        //jump only if the player is grounded and not in cooldown
        if (isGrounded)
        {
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
                float reward = 0;
                int action = 0;
                float boxDist = Player.transform.position.x - Box1.transform.position.x;
                float obst2Dist = Player.transform.position.x - Obstacle2.transform.position.x;
                bool fellDown = Player.transform.position.y - Floor.transform.position.y < 0;
                float rotationY = Player.transform.rotation.y;
                float height = Player.transform.position.y - initialPlayerPosition.y;

                float[] state = { boxDist, obst2Dist, height, Time.time - totalTime };

                if (Random.value < epsilon)
                    action = Random.Range(0, 1);
                else
                    action = ArgsMaxIndex(model.Predict(state));

                //if (action == 1)
                //    RotateRight();
                //else if (action == 2)
                //    RotateLeft();
                if(action == 1)
                    JumpPlayer();

                Debug.Log("Train with: Forward: " + state[0] + " Rotation: " + state[1]);

                if (fellDown)
                {
                    reward = -1f;
                    failedCount++;
                    done = true;
                }

                if (hitBox)
                {
                    reward = 0.05f;
                    hitBox = false;
                }

                //check for goal and obstacle conditions
                if (hitGoal)
                {
                    reward = (1 - Time.time - totalTime / 10) + 0.05f;
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
                if (reward != 0)
                {
                    float maxQValueNext = ArgsMax(model.FeedForward(state));
                    float qTarget = reward + gamma * maxQValueNext;

                    float[] qValues = model.FeedForward(state);
                    qValues[action] = qTarget;

                    model.Train(state, qValues, 0.05f);

                    overviewDisplay.text = $"Epochs: {trainedEpochs++}\nReward: {reward}\nPassed: {passedCount}\nFailed: {failedCount}\nTarget: {qTarget}\nQNext: {maxQValueNext}\nAction: {action}\nP/F: {passedFailedRatio}";

                    if(epsilon > epsilonMin)
                        epsilon -= epsilonReductionRate;

                    epsilonDisplay.text = $"Epsilon: {epsilon}";

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
