using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System.IO;

namespace image_recognition_Csharp
{
public static class Image
{
    public static double[,] LoadImage(string path)
    {
        Image<Gray8> image = SixLabors.ImageSharp.Image.Load<Gray8>(path);
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

    public static void SaveImage(string path, double[,] image)
    {
        int width = image.GetLength(0);
        int height = image.GetLength(1);
        byte[,] Byte_img = new byte[image.GetLength(0),image.GetLength(1)];
        for(int row=0;row<image.GetLength(0);row++)
        {
            for(int col=0;col<image.GetLength(1);col++)
            {
                Byte_img[row,col]=(byte)image[row,col];
            }
        }
        Image<Gray8> result = new Image<Gray8>(width, height);
        for (int i = 0; i < width; i++)
        {
            for (int j = 0; j < height; j++)
            {
                result[i, j] = new Gray8(Byte_img[i, j]);
            }
        }
        result.SaveAsPng(new FileStream(path, FileMode.Create));
    }

    }
}
