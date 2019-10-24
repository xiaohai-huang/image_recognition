using System;
using System.Collections.Generic;

namespace image_recognition_Csharp
{
    class Program
    {
        static void Main(string[] args)
        {
            
            
            // create a dictionary to store data and label of train_data
            // key = data
            // value = label
            Console.WriteLine("start");
            string MINST_FILE_PATH = @"mnist_train.csv";
            double[][] data = Image.Get_data(MINST_FILE_PATH);//row = label  ,, each row has 784 columns which are pixel values 
            int[] labels = Image.Get_labels(MINST_FILE_PATH);
            Dictionary<Matrix, int> train_data = new Dictionary<Matrix, int>();
            int index = 0;
            foreach (double[] row in data)
            {
                train_data.Add(new Matrix(one_D_array: row), labels[index]);
                index++;

            }
            
            // some variables, hyperparameter
            int num_of_classes = 10;
            int size_of_image = 28 * 28;
            Matrix Bias = new Matrix(row: num_of_classes, col: 1);
            Bias = Bias.Set_num(0.3);
            double delta = 1.0; // safe margin in the max function
            double step_size = 0.0001;
            int batch_size = 5;

            // get X_train, Y_train
            Matrix X_train=new Matrix(row:size_of_image,col:1);

            Matrix Y_train = new Matrix(row:batch_size,1);

            int row_index=0;
              foreach(var img_label in train_data)
            {
                Matrix input_data = img_label.Key.Reshape(1);
                X_train = X_train.Concatenate(input_data);
                Y_train[row_index]=img_label.Value;
                row_index++;
                if(row_index==batch_size)
                {break;}
            }
            X_train=X_train.Remove_Column(0);
            
            // initialize a random W
            Matrix W = new Matrix(num_of_classes, size_of_image);
            W = W.Set_num(0.03);

            Matrix gradient = ML.Eval_Numerical_Gradient(X_train,Y_train,Bias,W);

            double loss_orginal = ML.Get_Full_SVM_Loss(X_train,Y_train,Bias,W);
            Console.WriteLine($"orginal loss is {loss_orginal}");
            
            int time_tried = 0;
            while (true)
            {
                Matrix weights_grad = ML.Eval_Numerical_Gradient(X_train,Y_train,Bias,W);
                W= W + (-step_size*weights_grad);

                double loss_new = ML.Get_Full_SVM_Loss(X_train,Y_train,Bias,W);
                Console.WriteLine($"new loss is {loss_new}");

                string text_to_write = W.Return_String();

                string W_result_path = @"W_result.txt";

                System.IO.File.WriteAllText(W_result_path, $"{text_to_write}\nloss is {loss_new}\nTried times: {time_tried}");
                time_tried++;
                
                if(loss_new<1){break;}
            }
            
           
            

        }
    }
}
