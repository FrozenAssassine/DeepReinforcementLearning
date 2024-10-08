using JetBrains.Annotations;
using System.Collections;
using System.Linq;
using TMPro;
using UnityEditorInternal;
using UnityEngine;
using UnityEngine.UIElements;

public class Agent3 : MonoBehaviour
{
    [SerializeField] float playerSpeed = 5;
    [SerializeField] float jumpHeight = 5;
    [SerializeField] float epsilonReductionRate = 0.001f;
    [SerializeField] float epsilon = 1f;
    [SerializeField] float epsilonMin = 0.01f;
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
    int totalStates => Boxes.Length + Obstacles.Length + 1;
    int touchedBlockCount = 0;
    bool speedyMode = false;

    void Start()
    {
        model = NetworkBuilder.Create()
            .Stack(new InputLayer(totalStates)) //obst1 distance, obst2 distance, rotationY
            .Stack(new DenseLayer(100, ActivationType.Sigmoid))
            .Stack(new OutputLayer(4, ActivationType.Sigmoid))
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
        Player.transform.Translate(Vector3.left * ((speedyMode ? 1 : 0) + playerSpeed) * Time.deltaTime);
    }
    void JumpPlayer()
    {
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

                float[] state = new float[totalStates];

                bool fellDown = Player.transform.position.y - Floor.transform.position.y < 0;
                float rotationY = Player.transform.rotation.y;
                float height = Player.transform.position.y - initialPlayerPosition.y;

                for (int i = 0; i < Boxes.Length; i++)
                {
                    state[i] = Player.transform.position.x - Boxes[i].transform.position.x;
                }

                for (int i = 0; i < Obstacles.Length; i++)
                {
                    state[Boxes.Length + i] = Player.transform.position.x - Obstacles[i].transform.position.x;
                }
                state[totalStates - 1] = height;

                Debug.Log(string.Join(", ", state));

                if (Random.value < epsilon)
                {
                    action = Random.Range(0, 4);
                }
                else
                    action = ArgsMaxIndex(model.Predict(state));

                if (action == 1)
                    JumpPlayer();
                else if (action == 2)
                    speedyMode = true;
                else if (action == 3)
                    speedyMode = false;

                if (fellDown)
                {
                    reward = -1f;
                    failedCount++;
                    done = true;
                }

                if (hitBox)
                {
                    touchedBlockCount++;
                    reward = 0.02f * touchedBlockCount;
                    hitBox = false;
                }

                //check for goal and obstacle conditions
                if (hitGoal)
                {
                    //lower reward if it took longer:
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

                    if (epsilon > epsilonMin)
                        epsilon = Mathf.Clamp(epsilon - epsilonReductionRate, 0, 1);

                    epsilonDisplay.text = $"Epsilon: {epsilon}";

                    totalTime = Time.time;
                    touchedBlockCount = 0;
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
