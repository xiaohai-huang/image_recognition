using System;

namespace image_recognition_Csharp
{
    class Program
    {
        static void Main(string[] args)
        {


            //double[,] Scores = { { 13 },
            //                     { -7 },
            //                     {11 } };
            //Matrix matrix_Scores =new Matrix(Scores);
            //int correct_class_index = 0;
            //double delta = 10;

            //double loss = ML.Get_SVM_Loss(matrix_Scores, correct_class_index, delta);

            //Console.WriteLine(loss);





            //double[,] W =
            //{
            //    {0.2,-0.5,0.1,2.0 },
            //    {1.5,1.3,2.1,0.0 },
            //    {0,0.25,0.2,-0.3 }
            //};

            //double[,] Cat_image =
            //{
            //    { 56,231},
            //    {24,2 }
            //};

            //double[,] Bias =
            //{
            //    {1.1 },
            //    {3.2 },
            //    {-1.2 }
            //};

            //Matrix matrix_W = new Matrix(W);
            //Matrix matrix_Cat = new Matrix(Cat_image);// this has to be stretch to one column array
            //Matrix matrix_Bias = new Matrix(Bias);

            //Matrix reshapped_cat = matrix_Cat.Reshape(col: 1);

            //Matrix Scores = ML.Get_Scores(reshapped_cat, matrix_W,matrix_Bias);

            //double loss = ML.Get_SVM_Loss(Scores, 0, 1);

            //Console.WriteLine(loss);

            double[,] W_ =
            {
                {0.01,-0.05,0.1,0.05 },
                {0.7,0.2,0.05,0.16 },
                {0.0,-0.45,-0.2,0.03 },

            };

            double[,] input_ =
            {
                {-15 },
                {22 },
                {-44},
                {56 }
            };

            double[,] bias_ =
            {
                {0.0 },
                {0.2 },
                {-0.3 }
            };

            int y = 2;
            Matrix W = new Matrix(W_);
            Matrix input = new Matrix(input_);
            Matrix bias = new Matrix(bias_);
            Matrix Scores = ML.Get_Scores(input,W,bias);

            double loss = ML.Get_SVM_Loss(Scores, y, 1);
            Console.WriteLine(loss);


        }
    }
}
