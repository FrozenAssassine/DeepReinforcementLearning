using UnityEngine;
using UnityEngine.Assertions.Must;

public class Agent1 : MonoBehaviour
{
    float alpha = 0.1f;  // Learning rate
    float gamma = 0.9f;  // Discount factor
    int numEpisodes = 1000;
    float epsilon = 0.1f;  // Exploration rate

    NNModel model;

    void Start()
    {
        model = NetworkBuilder.Create()
            .Stack(new InputLayer(2))
            .Stack(new DenseLayer(4, ActivationType.Sigmoid))
            .Stack(new OutputLayer(2, ActivationType.Sigmoid))
            .Build();

        for (int episode = 0; episode < numEpisodes; episode++)
        {
            // Initialize the environment and state
            float distanceToObstacle = Random.Range(0.0f, 10.0f);  // Random initial distance
            float velocity = Random.Range(0.0f, 5.0f);  // Random initial velocity
            bool done = false;

            while (!done)
            {
                float[] state = { (float)distanceToObstacle, (float)velocity };

                // Choose action using epsilon-greedy policy
                int action;
                if (Random.Range(0.0f, 1.0f) < epsilon)
                {
                    // Exploration: Choose a random action
                    action = Random.Range(0, 2);
                }
                else
                {
                    // Exploitation: Choose the action with the highest Q-value
                    float[] res = model.FeedForward(state);
                    action = (res[1] > res[0]) ? 1 : 0;
                }

                // Perform the action and observe the reward and next state
                float reward = SimulateEnvironment(action, ref distanceToObstacle, ref velocity, out done);

                // Determine the maximum Q-value for the next state
                float[] qValuesNext = model.FeedForward(new float[] { (float)distanceToObstacle, (float)velocity });
                float maxQValueNext = Mathf.Max(qValuesNext[0], qValuesNext[1]);

                // Calculate the target Q-value
                float qTarget = reward + gamma * maxQValueNext;

                // Train the neural network to approximate Q(s, a)
                float[] qValues = model.FeedForward(state);
                qValues[action] = qTarget; // Update Q-value for the chosen action
                model.Train(state, qValues, 0.01f);
            }
        }
    }
}