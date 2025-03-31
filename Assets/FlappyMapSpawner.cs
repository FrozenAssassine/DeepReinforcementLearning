using UnityEngine;

public class FlappyMapSpawner : MonoBehaviour
{
    public GameObject obstaclePrefab; // Reference to the cube prefab
    public float spawnRate = 2f; // Time between spawns
    public float spawnOffset = 10f; // Distance between each obstacle
    public float minHeight = -2f; // Min height for the obstacle spawn
    public float maxHeight = 2f; // Max height for the obstacle spawn

    private float spawnTimer = 0f;
    private Vector3 lastSpawnPosition;

    void Start()
    {
        lastSpawnPosition = transform.position;
    }

    void Update()
    {
        spawnTimer += Time.deltaTime;

        // Spawn obstacles at a set rate
        if (spawnTimer >= spawnRate)
        {
            spawnTimer = 0f;
            SpawnObstacle();
        }
    }

    void SpawnObstacle()
    {
        // Random height for the obstacle
        float randomHeight = Random.Range(minHeight, maxHeight);

        // Create a new obstacle to the right of the last one
        Vector3 spawnPosition = new Vector3(lastSpawnPosition.x + spawnOffset, randomHeight, 0);
        Instantiate(obstaclePrefab, spawnPosition, Quaternion.identity);

        // Update the last spawn position
        lastSpawnPosition = spawnPosition;
    }
}
