using System;
using System.Collections.Generic;
using System.Text;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.PixelFormats;
using System.IO;

namespace image_recognition_Csharp
{
    class Image
    {
       
        public static void SaveImage(string path, double[,] double_image)
        {
            byte[,] image=new byte[double_image.GetLength(0),double_image.GetLength(1)];
            for (int x = 0; x < double_image.GetLength(0); x++)
            {
                for(int y = 0; y < double_image.GetLength(1); y++)
                {
                    image[x, y] = (byte)double_image[y, x];
                }
            }
            
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


        // for minist dataset
        public static int[] Get_labels(string file_path)
        {
            string[] lines = System.IO.File.ReadAllLines(file_path);
            int[] labels=new int[lines.Length];
            int index = 0;

            foreach(string line in lines)
            {
                string label = line.Substring(0,1);
                labels[index]=int.Parse(label);
                index = index + 1;
            }
            return labels;

        }

        public static double[][] Get_data(string file_path)
        {
            string[] lines = System.IO.File.ReadAllLines(file_path);
            double[,] data=new double[lines.Length,784];
            double[][] jagged_data = new double[lines.Length][];

            
            for (int row = 0; row < lines.Length; row++)
            {
                string line = lines[row];
                string pixel_value_str = line.Substring(2);

                string[] pixels_array = pixel_value_str.Split(",");
                for (int col = 0; col < pixels_array.Length; col++)
                {

                    data[row, col] = int.Parse(pixels_array[col]);
                    
                }
                

            }

            

            for (int row = 0; row < data.GetLength(0); row++)
            {
                double[] one_D_array = new double[data.GetLength(1)];

                for (int col = 0; col < data.GetLength(1); col++)
                {
                    one_D_array[col] = data[row, col];
                }
                jagged_data[row] = one_D_array;
            }
            return jagged_data;

        }
        

        
    }
}
