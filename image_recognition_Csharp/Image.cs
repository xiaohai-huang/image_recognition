
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System.IO;

namespace image_recognition_Csharp
{
public static class MRI
{
    public static double[,] LoadImage(string path)
    {
        Image<Gray8> image = Image.Load<Gray8>(path);
        double[,] result = new double[image.Width, image.Height];
        for (int i = 0; i < image.Width; i++)
        {
            for (int j = 0; j < image.Height; j++)
            {
                result[i, j] = image[i, j].PackedValue;
            }
        }
        return result;
    }

    static void SaveImage(string path, byte[,] image)
    {
        int width = image.GetLength(0);
        int height = image.GetLength(1);
        Image<Gray8> result = new Image<Gray8>(width, height);
        for (int i = 0; i < width; i++)
        {
            for (int j = 0; j < height; j++)
            {
                result[i, j] = new Gray8(image[i, j]);
            }
        }
        result.SaveAsPng(new FileStream(path, FileMode.Create));
    }

    }
}
