using System;

namespace image_recognition_Csharp
{
    class Program
    {
        static void Main(string[] args)
        {
            double[,] W =
            {
                {0.2,-0.5,0.1,2.0 },
                {1.5,1.3,2.1,0.0 },
                {0,0.25,0.2,-0.3 }
            };

            double[,] Cat_image =
            {
                { 56,231},
                {24,2 }
            };

            double[,] Bias =
            {
                {1.1 },
                {3.2 },
                {-1.2 }
            };

            Matrix matrix_W = new Matrix(W);
            Matrix matrix_Cat = new Matrix(Cat_image);
            Matrix matrix_Bias = new Matrix(Bias);

            Matrix reshapped_cat = matrix_Cat.Reshape(col: 1);

            Matrix Scores = (matrix_Bias)+ matrix_W.Dot(reshapped_cat);

            Scores.Display();
        }
    }
}
