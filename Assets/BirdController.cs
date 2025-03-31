using UnityEngine;

public class BirdController : MonoBehaviour
{
    public float gravity = -9.81f;  // Gravity for the bird's falling
    public float jumpForce = 10f;   // Force applied when the bird "jumps"
    public float raycastDistance = 10f; // Distance for raycast checks
    public LayerMask obstacleLayer;  // Layer for obstacles (cubes)

    private Rigidbody2D rb;
    private Vector2 velocity;

    void Start()
    {
        rb = GetComponent<Rigidbody2D>();
        velocity = rb.velocity;
    }

    void Update()
    {
        // Update the bird's velocity with gravity
        velocity.y += gravity * Time.deltaTime;
        rb.velocity = velocity;
    }

    // Function to calculate the bird's velocity
    public float GetVelocity()
    {
        return rb.velocity.y;
    }

    // Function to calculate the distance to the next top obstacle using raycast
    public float GetDistanceToTopObstacle()
    {
        RaycastHit2D hit = Physics2D.Raycast(transform.position, Vector2.up, raycastDistance, obstacleLayer);

        if (hit.collider != null)
        {
            // Return the distance to the top obstacle (positive value)
            return hit.distance;
        }
        else
        {
            // If no obstacle found, return a large value
            return float.MaxValue;
        }
    }

    // Function to calculate the distance to the next bottom obstacle using raycast
    public float GetDistanceToBottomObstacle()
    {
        RaycastHit2D hit = Physics2D.Raycast(transform.position, Vector2.down, raycastDistance, obstacleLayer);

        if (hit.collider != null)
        {
            // Return the distance to the bottom obstacle (positive value)
            return hit.distance;
        }
        else
        {
            // If no obstacle found, return a large value
            return float.MaxValue;
        }
    }

    public void Flap()
    {
        rb.velocity = new Vector2(rb.velocity.x, jumpForce);
    }
}

