using System;
using System.Collections.Generic;

namespace image_recognition_Csharp
{
    class Program
    {
        public static string image_root = @"D:\Stanford\image_recognition\MRI";

        public static void Main()
        {
            // get the data and labels
            string l_img_1 ="IXI536-Guys-1059-T1";
            string l_img_2 = "IXI539-Guys-1067-T1";

            string r_img_1="IXI625-Guys-1098-T1";
            string r_img_2="IXI482-HH-2178-T1";

            Image.LoadImage($@"{image_root}\{l_img_1}.png");


            Matrix W = new Matrix(row:2,col:316*256).Set_num(0.3);

            Matrix Bias = new Matrix(row:2,col:1).Set_num(0.5);

            // while(true)
            // {
            //     Matrix grad=ML.Eval_Numerical_Gradient()
            // }


        }
    }
}
