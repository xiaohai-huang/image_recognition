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

            double[,] l_img_1_arr = MRI.LoadImage($@"{image_root}\{l_img_1}.png");
            double[,] l_img_2_arr = MRI.LoadImage($@"{image_root}\{l_img_2}.png");
            double[,] r_img_1_arr = MRI.LoadImage($@"{image_root}\{r_img_1}.png");
            double[,] r_img_2_arr = MRI.LoadImage($@"{image_root}\{r_img_2}.png");
            

            double[,] Y_train_arr = 
            {
                {0},
                {1},
            };
            Matrix Y_train = new Matrix(Y_train_arr);

            Matrix L_1=new Matrix(l_img_1_arr);
            Matrix L_2 = new Matrix(l_img_2_arr);

            Matrix R_1 = new Matrix(r_img_1_arr);
            Matrix R_2 = new Matrix(r_img_2_arr);

            L_1=L_1.Reshape(1);
            L_2=L_2.Reshape(1);
            R_1=R_1.Reshape(1);
            R_2= R_2.Reshape(1);

            Matrix X_train = L_1;
            X_train=X_train.Concatenate(R_2);

            Matrix W = new Matrix(row:2,col:316*256).Set_num(0.3);

            Matrix Bias = new Matrix(row:2,col:1).Set_num(0.5);

            
           
            while(true)
            {
                Matrix grad=ML.Eval_Numerical_Gradient(X_train,Y_train,Bias,W);
                W+=-0.001*grad;
                double loss = ML.Get_Full_SVM_Loss(X_train,Y_train,Bias,W);
                Console.WriteLine(loss);

            }


        }
    }
}
