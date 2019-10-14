using System;

namespace image_recognition_Csharp
{
    class Program
    {
        static void Main(string[] args)
        {
            double[,] Weight =
            {
                { 1.00,2.00},
                {2.00, -4.00 },
                {3.00,-1.00 }
            };

            Matrix W = new Matrix(Weight);

            double regularization_loss = ML.Get_Regularization_Loss(W, 0.50119);
            Console.WriteLine(regularization_loss);


        }
    }
}
