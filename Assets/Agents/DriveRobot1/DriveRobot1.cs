using System.Collections;
using System.Collections.Generic;
using System.Data.Common;
using System.Linq;
using TMPro;
using UnityEngine;

public class DriveRobot1 : MonoBehaviour
{
    [SerializeField] WheelCollider leftWheelCollider;
    [SerializeField] WheelCollider rightWheelCollider;
    [SerializeField] float maxMotorTorque = 200f;
    [SerializeField] float maxTurnTorque = 150f;
    [SerializeField] float brakeForce = 5000f;
    [SerializeField] float ultrasonicMaxDistance = 2;
    [SerializeField] TMP_Text text;
    [SerializeField] int epochs = 1000;
    [SerializeField] Vector3 ultrasonicShift;
    float epsilon = .02f;
    bool fellDown = false;
    Vector3 initialPlayerPosition;
    Quaternion initialPlayerRotation;
    NNModel model;
    void OnDrawGizmos()
    {
        Vector3 raycastOrigin = transform.TransformPoint(ultrasonicShift);

        Gizmos.color = Color.blue;
        Gizmos.DrawRay(raycastOrigin, transform.forward * ultrasonicMaxDistance);
    }

    private void Start()
    {
        model = NetworkBuilder.Create()
            .Stack(new InputLayer(1))
            .Stack(new DenseLayer(100, ActivationType.Sigmoid))
            .Stack(new OutputLayer(2))
            .Build(false);

        initialPlayerPosition = this.transform.position;
        initialPlayerRotation = this.transform.rotation;

        StartCoroutine(TrainModel());
    }

    void ResetPlayer()
    {
        this.transform.position = initialPlayerPosition;
        this.GetComponent<Rigidbody>().velocity = Vector3.zero;
        this.transform.rotation = initialPlayerRotation;
    }

    IEnumerator TrainModel()
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            bool isEpisodeDone = false;

            while (!isEpisodeDone)
            {
                float reward = 0;

                var ultrasonicRange = MeasureUltrasonicSensor();

                float[] input = { ultrasonicRange};

                //predict move:
                float[] result;
                if (Random.value < epsilon)
                    result = new float[] { (Random.value * 2) - 1, (Random.value * 2) - 1};
                else
                    result = model.Predict(input);
                
                MoveRobot(result[0], result[1]);

                if (fellDown)
                {
                    isEpisodeDone = true;
                    fellDown = false;
                    reward = -1;
                }

                //do not collide with any walls => otherwise negative reward:
                if(ultrasonicRange <= .2f && ultrasonicRange > 0)
                {
                    isEpisodeDone = true;
                    reward = -0.5f;
                }

                //
                if(ultrasonicRange > 0.6f && ultrasonicRange > 0)
                {
                    reward = 0.0005f;
                }

                if (ultrasonicRange < 0.6f && ultrasonicRange > 0.3f)
                {
                    reward = 0.05f;
                }

                float[] output = model.Predict(input);
                float adjustmentFactor = 0.1f;
                float leftWheelSpeed = output[0];
                float rightWheelSpeed = output[1];

                if (reward != 0) 
                {
                    Debug.Log("Train with Reward: " + reward);
                    float[] adjustedTarget = new float[2];
                    if (reward > 0)
                    {
                        adjustedTarget[0] = leftWheelSpeed + adjustmentFactor * reward;
                        adjustedTarget[1] = rightWheelSpeed + adjustmentFactor * reward;
                    }
                    else
                    {
                        adjustedTarget[0] = leftWheelSpeed - adjustmentFactor * Mathf.Abs(reward);
                        adjustedTarget[1] = rightWheelSpeed - adjustmentFactor * Mathf.Abs(reward);
                    }

                    model.Train(input, adjustedTarget, 0.1f);
                }

                text.text = $"Epoch: {epoch}/{epochs}\n" +
                    $"Sensor Dist: {ultrasonicRange}\n"+
                    $"Epsilon: {epsilon}\n";

                yield return null;
            }
            //reduce epsilon over time for less randomness:
            epsilon -= 0.05f;
            if (epsilon < 0)
                epsilon = 0;

            ResetPlayer();
        }
    }

    private void Update()
    {
        
    }

    private void OnCollisionExit2D(Collision2D collision)
    {
        //player fell down:
        if (collision.gameObject.tag.Equals("Ground"))
        {
            //fellDown = true;
        }
    }

    private float MeasureUltrasonicSensor()
    {
        Vector3 raycastOrigin = transform.TransformPoint(ultrasonicShift);
        Vector3 direction = transform.forward;

        if (Physics.Raycast(raycastOrigin, direction, out RaycastHit hit, ultrasonicMaxDistance))
        {
            Debug.Log("Object detected: " + hit.collider.name + ", Distance: " + hit.distance.ToString("F2") + " meters");
            return hit.distance > ultrasonicMaxDistance ? -1 : hit.distance;
        }
        return -1;
    }
    private void MoveRobot(float move, float turn)
    {
        if (move != 0)
        {
            leftWheelCollider.motorTorque = move * maxMotorTorque;
            rightWheelCollider.motorTorque = move * maxMotorTorque;

            leftWheelCollider.brakeTorque = 0;
            rightWheelCollider.brakeTorque = 0;
        }
        else
        {
            leftWheelCollider.brakeTorque = brakeForce;
            rightWheelCollider.brakeTorque = brakeForce;
        }

        if (turn != 0)
        {
            leftWheelCollider.motorTorque = turn * maxTurnTorque;
            rightWheelCollider.motorTorque = -turn * maxTurnTorque;

            leftWheelCollider.brakeTorque = 0;
            rightWheelCollider.brakeTorque = 0;
        }
    }
}