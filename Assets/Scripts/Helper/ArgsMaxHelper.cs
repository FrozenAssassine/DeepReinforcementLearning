using System.Linq;

internal class ArgsMaxHelper
{
    public static int MaxIndex(float[] items)
    {
        return System.Array.IndexOf(items, items.Max());
    }
    public static float MaxValue(float[] items)
    {
        return items.Max();
    }

}