using System;
using System.IO;
using System.Collections.Generic;

namespace image_recognition_Csharp
{
    class Program
    {
       

        public static string image_root = @"MRI";
        public static string train_folder = @"D:\Stanford\image_recognition\image_recognition_Csharp\MRI\Training_set";
            
        public static void Train()
        {
            // Matrix test_matrix = new Matrix("web_demo_X.txt");
            // test_matrix=test_matrix.Remove_Row(1,3);
            // test_matrix.Display();
            Dictionary<string,Matrix> training_set = Matrix.Load_Image_Folder_Dict(train_folder);

            // create X matrix, and Y
            Matrix X = new Matrix(316*256,1);
            Matrix Y= new Matrix(training_set.Count,1);
            int index =0;
            foreach(var name_matrix in training_set)
            {
                X=X.Concatenate(name_matrix.Value);
                if(name_matrix.Key.Contains("left"))
                {
                    Y[index]=0;
                }
                else
                {Y[index]=1;}
                index++;
            }
            X=X.Remove_Column(0);


            // // train model
            // object[] W_b = ML.LogisticRegression.Train_model(X,Y.T,Save_Path:"test_trainMethod.txt");
            // Console.WriteLine(W_b[1]);



            // // update
            // Matrix W = new Matrix(316*256,1).Set_num(0.3);
            // double b = 2;
            // double m = X.Column;
            // double learning_rate = 0.003;

            // for (int i =0; i<10;i++)
            // {
            //     Matrix Z = W.T * X + b;
            //     Matrix A = ML.LogisticRegression.Sigmoid(Z);
            //     Matrix dZ = A-Y.T;
            //     Matrix dW = 1/m * X*dZ.T;
            //     double dB = 1/m * dZ.Sum();
                

            //     W = W - learning_rate*dW;
            //     b = b - learning_rate*dB;
            // }

            // W.SaveMatrix("W_T_logistic.txt");
        }
        public static void test()
        {
            // read W from file
            Matrix W = new Matrix("test_trainMethod.txt");

            // set up a random bias
            double b = -0.05;
            

            // key is the file name, value is image's pixel values in matrix
            Dictionary<string,Matrix> MRI_data=Matrix.Load_Image_Folder_Dict(image_root);
            
            // start testing
            foreach(var name_matrix in MRI_data)
            {

                Matrix Z = W.T*name_matrix.Value+b;
                Matrix result = ML.LogisticRegression.Sigmoid(Z);
                
                if(result[0]==1)
                {
                    Console.WriteLine($"{name_matrix.Key} is unsual");
                }
            }

        }

        public static void h()
        {
            // get X, Y data
            // Y has to be a 1-row matrix
            // X has to be stack into columns
            Dictionary<string,Matrix> training_set = Matrix.Load_Image_Folder_Dict(train_folder);

            // create X matrix, and Y
            Matrix X_train = new Matrix(316*256,1);
            Matrix Y_train= new Matrix(training_set.Count,1);
            int index =0;
            foreach(var name_matrix in training_set)
            {
                X_train=X_train.Concatenate(name_matrix.Value);
                if(name_matrix.Key.Contains("left"))
                {
                    Y_train[index]=0;
                }
                else
                {Y_train[index]=1;}
                index++;
            }
            X_train=X_train.Remove_Column(0);
            Y_train=Y_train.Reshape(1,Y_train.Row);
            
            Dictionary<string,object> results = ML.LogisticRegression.Model(
                X_train,Y_train,X_train,Y_train,print_cost:true,
                num_iterations:30);
            
            Matrix W = (Matrix)results["w"];
            W.SaveMatrix("test_trainMethod.txt");
            test();
        }

        public static void Main()
        {
            Matrix myMat = new Matrix(10, 10);
            myMat.Display();
            Console.WriteLine("git in Visual studio");
        }
    }
}
