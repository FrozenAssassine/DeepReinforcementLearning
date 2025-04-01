
// Environment manager class remains the same
using UnityEngine;
using UnityEngine.UIElements;

public class EnvironmentManager : MonoBehaviour
{
    public GameObject wallPrefab;
    public Vector2 arenaDimensions = new Vector2(20f, 20f);
    public float wallHeight = 2f;
    public int obstacleCount = 200;
    public int robots = 100;
    public GameObject robot;

    void Start()
    {
        CreateArena();

        for(int i = 0; i< robots; i++)
        {
            var r = Instantiate(robot);
            r.transform.position = new Vector3(Random.Range(-50, 50), 5f, Random.Range(-50, 50));
        }
    }

    void CreateArena()
    {
        // Create outer walls
        float wallThickness = 0.5f;

        // Bottom wall
        CreateWall(
            new Vector3(0, wallHeight / 2, -arenaDimensions.y / 2),
            new Vector3(arenaDimensions.x + wallThickness, wallHeight, wallThickness)
        );

        // Top wall
        CreateWall(
            new Vector3(0, wallHeight / 2, arenaDimensions.y / 2),
            new Vector3(arenaDimensions.x + wallThickness, wallHeight, wallThickness)
        );

        // Left wall
        CreateWall(
            new Vector3(-arenaDimensions.x / 2, wallHeight / 2, 0),
            new Vector3(wallThickness, wallHeight, arenaDimensions.y)
        );

        // Right wall
        CreateWall(
            new Vector3(arenaDimensions.x / 2, wallHeight / 2, 0),
            new Vector3(wallThickness, wallHeight, arenaDimensions.y)
        );

        // Add some obstacles inside the arena
        CreateObstacles(obstacleCount);
    }

    void CreateWall(Vector3 position, Vector3 size)
    {
        GameObject wall = GameObject.CreatePrimitive(PrimitiveType.Cube);
        wall.transform.position = position;
        wall.transform.localScale = size;
        wall.tag = "Wall";
    }

    void CreateObstacles(int count)
    {
        for (int i = 0; i < count; i++)
        {
            // Random position inside arena
            Vector3 position = new Vector3(
                Random.Range(-arenaDimensions.x / 2 + 1f, arenaDimensions.x / 2 - 1f),
                wallHeight / 2,
                Random.Range(-arenaDimensions.y / 2 + 1f, arenaDimensions.y / 2 - 1f)
            );

            // Random size
            Vector3 size = new Vector3(
                Random.Range(0.5f, 2f),
                wallHeight,
                Random.Range(0.5f, 2f)
            );

            CreateWall(position, size);
        }
    }
}