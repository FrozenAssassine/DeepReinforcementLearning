using JetBrains.Annotations;
using System.Collections;
using System.Linq;
using TMPro;
using UnityEditorInternal;
using UnityEngine;
using UnityEngine.UIElements;

public class Agent4 : MonoBehaviour
{
    [SerializeField] float playerSpeed = 5;
    [SerializeField] float playerRotationSpeed = 30;
    [SerializeField] float epsilonReductionRate = 0.001f;
    [SerializeField] float epsilon = .5f;
    [SerializeField] float epsilonMin = 0.01f;
    [SerializeField] float gamma = 0.9f;
    [SerializeField] int numEpisodes = 1000;
    [SerializeField] GameObject Player;
    [SerializeField] GameObject Goal; 
    [SerializeField] TMP_Text overviewDisplay;
    [SerializeField] TMP_Text epsilonDisplay;

    int trainedEpochs = 0;
    NNModel model;
    bool done = false;
    Vector3 initialPlayerPosition;
    Quaternion initialPlayerRotation;
    int passedCount = 0;
    int failedCount = 0;
    float passedFailedRatio = 0.0f;
    float totalTime = 0;
    float oldTimeScale = 0;
    bool hitGoal = false;

    void Start()
    {
        model = NetworkBuilder.Create()
            .Stack(new InputLayer(7))
            .Stack(new DenseLayer(100, ActivationType.Sigmoid))
            .Stack(new OutputLayer(3, ActivationType.Sigmoid))
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

    void ResetPlayer()
    {
        Player.transform.position = initialPlayerPosition;
        Player.transform.rotation = initialPlayerRotation;
        Player.GetComponent<Rigidbody>().velocity = Vector3.zero;
    }

    void WalkForward()
    {
        Player.transform.Translate(Vector3.forward * playerSpeed * Time.deltaTime);
    }

    void TurnLeft()
    {
        Player.transform.Rotate(0, playerRotationSpeed * Time.deltaTime, 0);
    }

    void TurnRight()
    {
        Player.transform.Rotate(0, -playerRotationSpeed * Time.deltaTime, 0);
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

                var pos = Player.transform.position;
                var rot = Player.transform.position;

                var goalDist = Vector3.Distance(Goal.transform.position, pos);
                float[] state = new float[] { pos.x, pos.y, pos.z, rot.x, rot.y, rot.z, goalDist};

                Vector3 directionToObstacle = (Goal.transform.position - Player.transform.position).normalized;
                float goalAngle = Vector3.Angle(Player.transform.forward, directionToObstacle);

                //action handling:
                if (Random.value < epsilon)
                    action = Random.Range(0, 3);
                else
                    action = ArgsMaxHelper.MaxIndex(model.Predict(state));

                if(action == 0)
                    WalkForward();
                else if(action == 1)
                    TurnLeft();
                else if (action == 2)
                    TurnRight();

                if(goalDist < 10)
                {
                    reward = ((11 - goalDist) + (180 - goalAngle)) / 5000;
                }

                if(goalAngle > 20 || goalAngle < -20)
                {
                    reward -= .1f;
                }

                if (hitGoal)
                    reward += 0.1f;

                //reward handling:
                reward -= 0.00005f; //reduce the longer it takes

                //handle falling:
                if (Player.transform.position.y < -1)
                {
                    reward = -1;
                    ResetPlayer();
                }

                //retrain the model when the reward is not 0
                if (reward != 0)
                {
                    Debug.Log(reward);
                    float maxQValueNext = ArgsMaxHelper.MaxValue(model.FeedForward(state));
                    float qTarget = reward + gamma * maxQValueNext;

                    float[] qValues = model.FeedForward(state);
                    qValues[action] = qTarget;

                    model.Train(state, qValues, 0.01f);

                    overviewDisplay.text = $"Epochs: {trainedEpochs++}\nReward: {reward}\nPassed: {passedCount}\nFailed: {failedCount}\nTarget: {qTarget}\nQNext: {maxQValueNext}\nAction: {action}\nP/F: {passedFailedRatio}";

                    if (epsilon > epsilonMin)
                        epsilon = Mathf.Clamp(epsilon - epsilonReductionRate, 0, 1);

                    epsilonDisplay.text = $"Angle: {goalAngle}\nDist{goalDist}";

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
            done = true;
            hitGoal = true;
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
