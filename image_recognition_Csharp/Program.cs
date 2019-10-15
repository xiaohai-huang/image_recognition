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
            double step_size = 0.1;

            // initialize a random W
            Matrix W = new Matrix(num_of_classes, size_of_image);
            W = W.Set_num(0.4);



         

            foreach (var item in train_data)
            {
                Console.WriteLine($"start label #{item.Value}");


                // stretch the image
                Matrix input_image = item.Key;
                input_image = input_image.Reshape(1);// reshape to 1 column


                int time_tried = 0;
                while (true)
                {
                    Matrix old_scores = ML.Get_Scores(input_image, W, Bias);
                    double old_loss = ML.Get_SVM_Loss(old_scores, item.Value, delta);
                    //
                    Matrix gradient = ML.Get_Numerical_Gradient(input_image, Bias, W, (int)item.Value);
                    W += -step_size * gradient;
                    //
                    
                    Matrix new_score = ML.Get_Scores(input_image, W, Bias);
                    double new_loss = ML.Get_SVM_Loss(new_score, (int)item.Value, delta);

                    Console.WriteLine($"#{time_tried}");
                    Console.WriteLine($"The higgest score is for label {new_score.Get_Max_index()}, which is {Math.Round(new_score.Get_Max())}");
                    Console.WriteLine($"the loss for label {item.Value} is '{Math.Round(new_loss)}' and the score for that is {Math.Round(new_score[item.Value])}");
                    Console.WriteLine($"The score increase rate is {Math.Round((new_score[item.Value] - old_scores[item.Value]) / old_scores[item.Value], 3)}%");
                    Console.WriteLine($"The decrease rate of loss is {Math.Round(-(new_loss - old_loss) / old_loss),5}%\n\n");
                    
                    string text_to_write = W.Return_String();
                    string W_result_path = @"W_result.txt";
                    System.IO.File.WriteAllText($"{W_result_path}_{item.Value}", $"{text_to_write}\n\nThis is for label {item.Value}\nTried times: {time_tried}");

                    time_tried++;

                    if (new_loss <= 10)
                    {
                        Console.WriteLine("The final loss is " + new_loss);
                        W.Display();
                        
                        Console.WriteLine("Finsihed!!!!!!!!!\n\n");
                       
                        break;
                    }
                }
            }

            

        }
    }
}
