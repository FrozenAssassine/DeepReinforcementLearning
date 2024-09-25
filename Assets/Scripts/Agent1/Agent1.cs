using System.Collections;
using System.Linq;
using TMPro;
using UnityEngine;

public class Agent1 : MonoBehaviour
{
    [SerializeField] GameObject Player;
    [SerializeField] GameObject Obstacle;
    [SerializeField] GameObject Goal;
    [SerializeField] TMP_Text epochDisplay;

    float playerSpeed = 5;
    float jumpHeight = 5;

    float gamma = 0.9f;
    int numEpisodes = 1000;
    float epsilon = 0.1f;

    bool hitGoal = false;
    bool hitObstacle = false;

    int trainedEpochs = 0;
    NNModel model;
    bool done = false;
    Vector3 initialPlayerPosition;

    float didNothingCount = 0;

    void Start()
    {
        //Time.timeScale = 10;
        model = NetworkBuilder.Create()
            .Stack(new InputLayer(2))                   // Input Layer with 2 inputs
            .Stack(new DenseLayer(8, ActivationType.Relu)) // First Hidden Layer with 8 neurons
            .Stack(new DenseLayer(4, ActivationType.Relu)) // Second Hidden Layer with 4 neurons
            .Stack(new OutputLayer(3, ActivationType.Softmax)) // Output Layer with 3 outputs
            .Build(false);

        initialPlayerPosition = Player.transform.position;

        StartCoroutine(TrainModel());
    }

    int ArgsMax(float[] items)
    {
        return System.Array.IndexOf(items, items.Max());
    }

    IEnumerator TrainModel()
    {
        for (int episode = 0; episode < numEpisodes; episode++)
        {
            bool isEpisodeDone = false;

            while (!isEpisodeDone)
            {
                // Calculate the current state based on the actual environment
                float distanceToObstacle = Player.transform.position.x - Obstacle.transform.position.x;
                float velocity = Player.GetComponent<Rigidbody>().velocity.x;
                float[] state = { distanceToObstacle, velocity };
                float reward = 0;

                // Choose action based on epsilon-greedy strategy
                int action;
                if (Random.value < epsilon)
                {
                    action = Random.Range(0, 3);  // Random action
                }

                else
                {
                    float[] res = model.FeedForward(state);
                    action = ArgsMax(res);
                    Debug.Log("Prediction: " + action);
                }

                if(Player.GetComponent<Rigidbody>().velocity.x == 0)
                {
                    didNothingCount++;
                    if (didNothingCount > 100)
                    {
                        didNothingCount = 0;
                        reward = -1;
                    }
                }

                // Perform actions based on the chosen action
                switch (action)
                {
                    case 0:
                        //do nothing:
                        break;
                    case 1:
                        MovePlayer();
                        break;
                    case 2: // Jump
                        JumpPlayer();
                        break;
                }


                if (hitGoal)
                {
                    reward = 1.0f;  // Positive reward for reaching the goal
                    hitGoal = false;  // Reset flag
                }
                else if (hitObstacle)
                {
                    reward = -1.0f;  // Negative reward for hitting an obstacle
                    hitObstacle = false;  // Reset flag
                }

                if (reward != 0)
                {
                    Debug.Log("REWARD: " + reward);

                    // Get Q-values for the next state
                    float[] qValuesNext = model.FeedForward(state);
                    float maxQValueNext = qValuesNext[0];

                    // Calculate target Q-value
                    float qTarget = reward + gamma * maxQValueNext;

                    // Get Q-values for the current state and update the chosen action
                    float[] qValues = model.FeedForward(state);
                    qValues[action] = qTarget;

                    // Train the model
                    model.Train(state, qValues, 0.1f);
                    epochDisplay.text = "Epochs: " + trainedEpochs++;

                    // End episode if done
                    if (done)
                    {
                        isEpisodeDone = true;
                        done = false;  // Reset flag for next episode
                    }
                }

                yield return null;  // Wait for the next frame
            }

            // Reset the player for the next episode
            ResetPlayer(initialPlayerPosition);
        }
    }

    void ResetPlayer(Vector3 homePosition)
    {
        // Reset player position and velocity
        Player.transform.position = homePosition;
        Player.GetComponent<Rigidbody>().velocity = Vector3.zero;
    }

    void MovePlayer()
    {
        // Move the player continuously to the left
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
            Debug.Log("HIT GOAL");
            hitGoal = true;
            done = true;
        }
        else if (collision.gameObject == Obstacle)
        {
            Debug.Log("HIT OBstacle");
            hitObstacle = true;
            done = true;
        }

        // Detect collision with ground (for jumping)
        if (collision.gameObject.CompareTag("Ground"))
        {
            isGrounded = true;
        }
    }

    void Update()
    {
        if (done)
        {
            done = false;  // Reset for next episode
            ResetPlayer(initialPlayerPosition);
        }
    }
}
