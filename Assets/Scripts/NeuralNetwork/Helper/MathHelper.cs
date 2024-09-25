using System.Linq;
using UnityEngine;

public class MathHelper
{
    public static float RandomFloat1_1()
    {
        return (Random.value * 2) - 1;
    }
    public static float RandomFloat0_1()
    {
        return Random.value;
    }

    public static int GetMaximumIndex(float[] items)
    {
        return System.Array.IndexOf(items, items.Max());
    }
}
