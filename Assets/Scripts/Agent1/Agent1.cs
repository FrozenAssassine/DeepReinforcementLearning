using System.Collections;
using System.Linq;
using TMPro;
using UnityEngine;

public class Agent1 : MonoBehaviour
{
    [SerializeField] GameObject Player;
    [SerializeField] GameObject Obstacle;
    [SerializeField] GameObject Goal;
    [SerializeField] GameObject Checkpoint;
    [SerializeField] TMP_Text epochDisplay;

    float playerSpeed = 5;
    float jumpHeight = 6;

    float gamma = 0.9f;
    int numEpisodes = 1000;
    float epsilon = 1f;

    bool hitGoal = false;
    bool hitObstacle = false;
    bool hitCheckPoint = false;

    int trainedEpochs = 0;
    NNModel model;
    bool done = false;
    Vector3 initialPlayerPosition;
    int successfull = 0;
    int failed = 0;

    void Start()
    {
        Time.timeScale = 2;
        model = NetworkBuilder.Create()
            .Stack(new InputLayer(2))
            .Stack(new DenseLayer(50, ActivationType.Softmax))
            .Stack(new OutputLayer(2, ActivationType.Softmax))
            .Build(false);

        initialPlayerPosition = Player.transform.position;

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

    void ReduceEpsilon()
    {
        if (epsilon > 0.1f)
            epsilon -= 0.005f;
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
                float velocity = Player.GetComponent<Rigidbody>().velocity.y;
                float[] state = { distanceToObstacle, velocity };
                float reward = 0;

                // Predict or generate random action
                int action;
                if (Random.value < epsilon)
                {
                    action = Random.Range(0, 2); // Random action
                }
                else
                {
                    var pred = model.FeedForward(state);
                    action = ArgsMaxIndex(pred);
                    Debug.Log(pred[0] + ":" + pred[1]);
                }

                // Execute the action
                if (action == 1) // Jump
                {
                    JumpPlayer();
                }

                // Check for goal and obstacle conditions
                if (hitGoal)
                {
                    reward = 1.0f;
                    successfull++;
                    epsilon = Mathf.Max(epsilon - 0.1f, 0.01f); // Clamp epsilon to a minimum
                    hitGoal = false;
                }
                else if (hitObstacle)
                {
                    reward = -1f;
                    failed++;
                    hitObstacle = false;
                }
                //else if (hitCheckPoint)
                //{
                //    Debug.Log("!Hit checkpoint");
                //    reward = 2;
                //    hitCheckPoint = false;
                //}

                // Retrain the model when the reward is not 0
                if (reward != 0)
                {
                    float maxQValueNext = ArgsMax(model.FeedForward(state));
                    float qTarget = reward + gamma * maxQValueNext;

                    float[] qValues = model.FeedForward(state);
                    qValues[action] = qTarget;

                    model.Train(state, qValues, 0.05f);

                    epochDisplay.text = $"Epochs: {trainedEpochs++}\nReward: {reward}\nPassed: {successfull}\nFailed: {failed}\nTarget: {qTarget}\nQNext: {maxQValueNext}";

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

    bool isGrounded;
    float jumpCooldown = 1.0f;
    float lastJumpTime;

    void JumpPlayer()
    {
        // Jump only if the player is grounded and not in cooldown
        if (isGrounded && Time.time - lastJumpTime > jumpCooldown)
        {
            Player.GetComponent<Rigidbody>().AddForce(Vector3.up * jumpHeight, ForceMode.Impulse);
            isGrounded = false;
            lastJumpTime = Time.time;
        }
    }

    void OnCollisionEnter(Collision collision)
    {
        // Detect collision with goal or obstacle
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

        // Detect collision with ground for jumping
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
