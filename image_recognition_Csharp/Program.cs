using System;
using System.Collections.Generic;

namespace image_recognition_Csharp
{
    class Program
    {
        public static string image_root = @"MRI";

        public static void Train_method()
        {
            Matrix W = new Matrix(row:2,col:316*256);
            
            // get the data and labels
            string l_img_1 ="IXI536-Guys-1059-T1";
            string l_img_2 = "IXI539-Guys-1067-T1";
            string l_img_3 = "IXI050-Guys-0711-T1";
            string l_img_4 = "IXI051-HH-1328-T1";
            string l_img_5 = "IXI085-Guys-0759-T1";
            string l_img_6 = "IXI128-HH-1470-T1";

            string r_img_1="IXI625-Guys-1098-T1";
            string r_img_2="IXI482-HH-2178-T1";

            double[,] l_img_1_arr = Image.LoadImage($@"{image_root}/{l_img_1}.png");
            double[,] l_img_2_arr = Image.LoadImage($@"{image_root}/{l_img_2}.png");
            double[,] l_img_3_arr = Image.LoadImage($@"{image_root}/{l_img_3}.png");
            double[,] l_img_4_arr = Image.LoadImage($@"{image_root}/{l_img_4}.png");
            double[,] l_img_5_arr = Image.LoadImage($@"{image_root}/{l_img_5}.png");
            double[,] l_img_6_arr = Image.LoadImage($@"{image_root}/{l_img_6}.png");
            


            double[,] r_img_1_arr = Image.LoadImage($@"{image_root}/{r_img_1}.png");
            double[,] r_img_2_arr = Image.LoadImage($@"{image_root}/{r_img_2}.png");
            

            double[,] Y_train_arr = 
            {
                {0},
                {0},
                {0},
                {0},
                {0},
                {0},
                {1},
                {1},
            };
            Matrix Y_train = new Matrix(Y_train_arr);

            Matrix L_1=  new Matrix(l_img_1_arr).Reshape(1);
            Matrix L_2 = new Matrix(l_img_2_arr).Reshape(1);
            Matrix L_3 = new Matrix(l_img_3_arr).Reshape(1);
            Matrix L_4 = new Matrix(l_img_4_arr).Reshape(1);
            Matrix L_5 = new Matrix(l_img_5_arr).Reshape(1);
            Matrix L_6 = new Matrix(l_img_6_arr).Reshape(1);

            Matrix R_1 = new Matrix(r_img_1_arr).Reshape(1);
            Matrix R_2 = new Matrix(r_img_2_arr).Reshape(1);

            

            Matrix X_train = L_1;
            X_train=X_train.Concatenate(L_2).Concatenate(L_3).Concatenate(L_4).Concatenate(L_5).Concatenate(L_6).Concatenate(R_1).Concatenate(R_2);
            
            Matrix Bias = new Matrix(row:2,col:1).Set_num(0.5);
            
           ML.Train_model(X_train,Y_train);

        }
        public static void Test_MRI()
        {
            // read W from file
            Matrix W = new Matrix("W.txt");

            // set up a random bias
            Matrix Bias = new Matrix(row:2,col:1).Set_num(0.5);

            // key is the file name, value is image's pixel values in matrix
            Dictionary<string,Matrix> MRI_data=Matrix.Load_Image_Folder_Dict(image_root);

            // start testing
            foreach(var name_matrix in MRI_data)
            {
                Matrix score = ML.Get_Scores(name_matrix.Value,W,Bias);
                
                // to see wether it is in index 0 or 1
                Matrix result = Matrix.Get_Max(score);
                if(result[0]!=0)
                {
                    Console.WriteLine($"{name_matrix.Key} is unsual");
                }
            }

        }
  
  
  
        public static Matrix X = new Matrix("web_demo_X.txt").T;
        public static Matrix Y = new Matrix("web_demo_Y.txt");
        public static Matrix Bias = new Matrix("web_demo_Bias.txt");
        public static double web_demo_Loss_func(Matrix W)
        {
            // double loss = ML.Get_Full_SVM_Loss(X,Y,Bias,W);
            double loss = ML.Get_Full_SVM_Loss_with_Regularization(X,Y,Bias,W,0.1);
            return loss;
        }
        public static void Main()
        {
            Matrix W = new Matrix("web_demo_W.txt");
            double loss = web_demo_Loss_func(W);
            Matrix grad = ML.Eval_Numerical_Gradient(web_demo_Loss_func,W);

            W = ML.Train_model(web_demo_Loss_func,W);
            W.Display();
            double acc = ML.Get_accuracy(X,Y,W);
        }
    }
}
