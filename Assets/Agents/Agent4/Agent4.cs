using System.Collections;
using System.Linq;
using TMPro;
using UnityEngine;

public class Agent4 : MonoBehaviour
{
    [SerializeField] float playerSpeed = 5;
    [SerializeField] float roatationSpeed = 20;
    [SerializeField] float epsilonReductionRate = 0.001f;
    [SerializeField] float epsilon = 1f;
    [SerializeField] float epsilonMin = 0.01f;
    [SerializeField] float gamma = 0.9f;
    [SerializeField] int numEpisodes = 1000;
    [SerializeField] GameObject Player;
    [SerializeField] GameObject Floor;
    [SerializeField] TMP_Text overviewDisplay;
    [SerializeField] TMP_Text epsilonDisplay;
    public float movementThreshold = 0.5f; // The distance within which the player is considered stationary
    public float idleTime = 2.0f;          // The time in seconds to be considered "idle"

    private Vector3 lastPosition;          // The last recorded position of the player
    private float idleTimer = 0.0f;        // Timer to track idle duration

    bool isGrounded = false;
    NNModel model;
    bool done = false;
    Vector3 initialPlayerPosition;
    Quaternion initialPlayerRotation;
    int totalStates = 2;

    public float raycastRange = 10.0f;

    void Start()
    {
        model = NetworkBuilder.Create()
            .Stack(new InputLayer(totalStates))
            .Stack(new DenseLayer(100, ActivationType.Sigmoid))
            .Stack(new OutputLayer(4, ActivationType.Sigmoid))
            .Build(false);

        initialPlayerPosition = Player.transform.position;
        initialPlayerRotation = Player.transform.rotation;
        lastPosition = transform.position;

        StartCoroutine(TrainModel());
    }


    private bool PlayerDidNotMoved()
    {
        float distanceMoved = Vector3.Distance(transform.position, lastPosition);

        if (distanceMoved <= movementThreshold)
        {
            idleTimer += Time.deltaTime;
        }
        else
        {
            idleTimer = 0.0f;
        }


        if (idleTimer >= idleTime)
        {
            done = true;
            lastPosition = transform.position;
            return true;
        }

        return false;
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

    void RoateLeft()
    {
        Player.transform.Rotate(new Vector3(0,-1,0) * roatationSpeed * Time.deltaTime);
    }
    void RotateRight()
    {
        Player.transform.Rotate(new Vector3(0, 1, 0) * roatationSpeed * Time.deltaTime);
    }

    void WalkForward()
    {
        Player.transform.Translate(Vector3.forward * playerSpeed * Time.deltaTime);
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

                float rotationY = Player.transform.rotation.y;
                state[0] = rotationY;

                Debug.Log(string.Join(", ", state));

                if (Random.value < epsilon)
                {
                    action = Random.Range(0, 4);
                }
                else
                    action = ArgsMaxIndex(model.Predict(state));

                if (action == 1)
                    WalkForward();
                else if (action == 2)
                    RoateLeft();
                else if (action == 3)
                    RotateRight();

                if (PlayerDidNotMoved())
                    reward = -1f;

                //raycasts:
                if (Physics.Raycast(transform.position, transform.forward, out RaycastHit hitInfo, raycastRange))
                {
                    if (hitInfo.collider.gameObject.tag.Equals("Obstacle"))
                    {
                        state[1] = hitInfo.distance;
                    }

                    if (hitInfo.collider.gameObject.tag.Equals("BadPlayer"))
                    {
                        state[2] = hitInfo.distance;
                        reward = 0.1f;
                    }
                }



                //retrain the model when the reward is not 0
                if (reward != 0)
                {
                    float maxQValueNext = ArgsMax(model.FeedForward(state));
                    float qTarget = reward + gamma * maxQValueNext;

                    float[] qValues = model.FeedForward(state);
                    qValues[action] = qTarget;

                    model.Train(state, qValues, 0.05f);

                    //overviewDisplay.text = $"Epochs: {trainedEpochs++}\nReward: {reward}\nPassed: {passedCount}\nFailed: {failedCount}\nTarget: {qTarget}\nQNext: {maxQValueNext}\nAction: {action}\nP/F: {passedFailedRatio}";

                    if (epsilon > epsilonMin)
                        epsilon = Mathf.Clamp(epsilon - epsilonReductionRate, 0, 1);

                    epsilonDisplay.text = $"Epsilon: {epsilon}";

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
        //if (collision.gameObject.tag.Equals("Obstacle"))
        //{
        //    hitObstacle = true;
        //    done = true;
        //}

        if (collision.gameObject.layer == 3) //ground
        {
            isGrounded = true;
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

    private void OnDrawGizmos()
    {
        Gizmos.color = Color.red;
        Gizmos.DrawRay(transform.position, transform.forward * raycastRange);
    }
}

