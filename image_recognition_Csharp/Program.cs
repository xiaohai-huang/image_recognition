﻿using System;
using System.Collections.Generic;

namespace image_recognition_Csharp
{
    class Program
    {
        static void test1()
        {
            // Matrix W = new Matrix(row:10,col:28*28);
            // W=W.Set_num(3);
            //  double[,] X_test_arr=
            // {   //1
            //     //{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,168,242,28,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,10,228,254,100,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,190,254,122,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,83,254,162,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,29,254,248,25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,29,255,254,103,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,29,254,254,109,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,29,254,254,109,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,29,254,254,109,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,29,255,254,109,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,29,254,254,109,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,29,254,254,63,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,29,254,254,28,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,29,254,254,28,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,29,254,254,35,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,29,254,254,109,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,212,254,109,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,203,254,178,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,155,254,190,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,32,199,104,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},

            //     //2
            //     {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,93,164,211,250,250,194,15,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,20,176,253,237,180,180,243,254,214,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,204,236,135,18,0,0,40,242,252,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,69,253,167,0,0,0,0,0,130,254,223,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,74,217,79,0,0,0,0,0,46,254,231,14,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,10,0,0,0,0,0,0,39,254,254,104,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,212,254,141,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,207,254,141,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7,215,254,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,39,254,254,56,0,0,20,67,124,39,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7,35,98,254,254,208,157,207,225,254,241,160,0,0,0,0,0,0,0,0,0,0,9,31,82,137,203,203,212,254,254,254,254,251,223,223,127,52,33,0,0,0,0,0,0,0,0,0,9,137,214,254,254,254,254,240,228,250,254,254,154,50,0,0,0,0,0,0,0,0,0,0,0,0,0,0,185,254,247,179,146,67,60,28,0,216,254,220,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,222,49,0,0,0,0,4,137,244,232,50,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,254,206,4,0,0,0,8,179,254,247,64,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,216,254,158,177,130,96,213,252,199,49,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,131,247,249,249,249,171,72,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}    
            // };
            // Matrix X_test = new Matrix(X_test_arr);
            // X_test =X_test.Reshape(28*28,X_test.Row);
            // Matrix scores = W*X_test;
            // scores.Display();
            double[,] a_arr=
            {
                {1,2,3},
                {4,5,6}
            };
            double[,] b_arr=
            {
                {7,8,9},
                {10,11,12}
            };
            Matrix A = new Matrix(a_arr);
            Matrix B = new Matrix(b_arr);
            Console.WriteLine(A.Return_String());
            A.Display();
            Console.WriteLine();
            A=A.Remove_Column(1);
            A.Display();
            Console.ReadLine();
        }
        static void Main()
        {
            // test loss function
            
            string file_path=@"mnist_train.csv";
            double[][] x_arr = Image.Get_data(file_path);
            double[] y_arr =Image.Get_labels(file_path);

            // sample 256 examples
            x_arr = Matrix.sample_training_data(x_arr,32);
            y_arr=Matrix.sample_training_data(y_arr,32);
                        


            // bias
            double[,] b_arr =
            {
                {0},
                {0.5},
                {2.2},
                {-0.5},
                {0},
                {0.5},
                {2.2},
                {-0.5},
                {0},
                {0.5},
            };

            double[][] X_test_arr=
            {   //1
                new double[]{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,168,242,28,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,10,228,254,100,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,190,254,122,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,83,254,162,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,29,254,248,25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,29,255,254,103,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,29,254,254,109,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,29,254,254,109,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,29,254,254,109,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,29,255,254,109,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,29,254,254,109,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,29,254,254,63,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,29,254,254,28,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,29,254,254,28,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,29,254,254,35,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,29,254,254,109,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,212,254,109,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,203,254,178,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,155,254,190,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,32,199,104,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},

                //2
                new double[]{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,93,164,211,250,250,194,15,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,20,176,253,237,180,180,243,254,214,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,204,236,135,18,0,0,40,242,252,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,69,253,167,0,0,0,0,0,130,254,223,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,74,217,79,0,0,0,0,0,46,254,231,14,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,10,0,0,0,0,0,0,39,254,254,104,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,212,254,141,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,207,254,141,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7,215,254,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,39,254,254,56,0,0,20,67,124,39,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7,35,98,254,254,208,157,207,225,254,241,160,0,0,0,0,0,0,0,0,0,0,9,31,82,137,203,203,212,254,254,254,254,251,223,223,127,52,33,0,0,0,0,0,0,0,0,0,9,137,214,254,254,254,254,240,228,250,254,254,154,50,0,0,0,0,0,0,0,0,0,0,0,0,0,0,185,254,247,179,146,67,60,28,0,216,254,220,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,222,49,0,0,0,0,4,137,244,232,50,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,254,206,4,0,0,0,8,179,254,247,64,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,216,254,158,177,130,96,213,252,199,49,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,131,247,249,249,249,171,72,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},    

                //6 
                new double[]{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,34,169,250,40,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,58,242,221,143,17,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,75,247,143,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,37,245,184,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,192,200,14,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,139,247,28,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7,231,183,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,125,243,50,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,195,184,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,61,251,41,0,0,0,64,43,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,152,210,7,0,96,237,254,247,107,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,250,84,0,6,223,84,13,87,246,72,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,43,254,80,0,56,151,0,0,0,147,193,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,67,254,41,0,13,19,0,0,0,42,253,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,67,254,13,0,0,0,0,0,0,14,253,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,68,255,13,0,0,0,0,0,0,77,240,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,67,254,13,0,0,0,0,0,5,181,147,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,25,229,105,0,0,0,0,5,156,213,20,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,107,246,105,14,49,95,217,209,27,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,107,246,253,253,240,130,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},

                //9
                new double[]{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,18,145,255,254,249,66,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,59,253,253,253,253,253,242,54,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,97,223,253,212,101,82,250,247,70,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,38,225,242,154,15,0,0,193,224,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,38,225,253,170,0,0,0,93,252,238,42,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,38,224,253,182,11,0,0,0,162,253,224,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,89,253,248,57,0,0,0,47,242,253,111,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,101,253,140,0,0,6,24,165,253,236,47,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,132,253,251,160,160,182,253,253,253,104,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,69,247,253,253,253,253,253,253,253,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,52,61,165,132,230,253,253,166,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,206,253,253,97,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,42,253,253,167,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,162,253,253,98,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,93,253,253,194,17,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,163,253,253,59,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,37,253,253,180,17,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,218,253,220,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,243,253,165,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,243,242,50,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},

            };
            
            
            Matrix X_train = new Matrix(x_arr);
            Matrix Y_train = new Matrix(y_arr);//change the constructor
            Y_train=Y_train.Reshape(1);


            Matrix X_test = new Matrix(X_test_arr);
            

            Matrix W = new Matrix(row:10,col:28*28);
            W=W.Set_num(3);
            Matrix Bias = new Matrix(b_arr);
           
           
            // update W
            int time_tried=1;
            while (true)
            {
                Matrix grad = ML.Eval_Numerical_Gradient(X_train, Y_train, Bias, W);
                W += -0.0001 * grad;
                double loss_new = ML.Get_Full_SVM_Loss(X_train, Y_train, Bias, W);
                Console.WriteLine("the loss is  "+loss_new);
                
                string text_to_write = W.Return_String();
                string W_result_path = @"W_result.txt";
                System.IO.File.WriteAllText(W_result_path, $"{text_to_write}\nloss is {loss_new}\nTried times: {time_tried}");
                time_tried++;


                if(loss_new<1){break;}
                
            }
            
            // predict
            Matrix scores = W.Dot(X_test);
            Matrix Y_predict = Matrix.Get_Max(scores);
            scores.Display();

            Console.WriteLine();
            Y_predict.Display();
            Console.ReadLine();
        

        }
    }
}
