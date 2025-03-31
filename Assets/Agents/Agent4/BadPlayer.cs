using System.Collections;
using UnityEngine;

public class BadPlayer : MonoBehaviour
{
    public float speed = 3.0f;
    public float walkRadius = 10.0f;
    public float waitTime = 2.0f;

    private Vector3 targetPosition;
    private bool waiting = false;

    private void Start()
    {
        SetNewTargetPosition();
    }

    private void Update()
    {
        if (!waiting)
        {
            transform.position = Vector3.MoveTowards(transform.position, targetPosition, speed * Time.deltaTime);

            if (Vector3.Distance(transform.position, targetPosition) < 0.1f)
            {
                StartCoroutine(WaitBeforeNewTarget());
            }
        }
    }

    private IEnumerator WaitBeforeNewTarget()
    {
        waiting = true;
        yield return new WaitForSeconds(waitTime);
        SetNewTargetPosition();
        waiting = false;
    }

    private void SetNewTargetPosition()
    {
        Vector2 randomDirection = Random.insideUnitCircle * walkRadius; // Get a random 2D direction
        targetPosition = new Vector3(randomDirection.x, transform.position.y, randomDirection.y); // Set the target position
    }
}