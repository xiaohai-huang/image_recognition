using System;


namespace image_recognition_Csharp
{
    static class ML
    {
 

        // score function
        public static Matrix Get_Scores(Matrix input_data, Matrix W, Matrix Bias)
        {
            Matrix Scores = W * input_data + Bias;

            return Scores;

        }

        // SVM loss function calculate the loss of the given image
        // data loss
        public static double Get_SVM_Loss(Matrix Scores, int correct_class_index, double delta)
        {
            double loss = 0;
            for (int x = 0; x < Scores.Row; x++)
            {
                // do not iterate the corret class score
                if (x == correct_class_index)
                {
                    continue;
                }
                else
                {

                    // the socre of the correct class is higher than the other score by at least a margin of delta
                    // add loss 0

                    // [incorrect score + delat] should be not be greater than the correct one 
                    // if it does, we accumulate the loss
                    if (Scores[correct_class_index] > Scores[x]+delta)
                    {
                        loss = loss + 0;
                    }
                    // if any class has a socre that higher than correct class's score + delta 
                    // accumulate the loss
                    else
                    {
                        loss = loss + (Scores[x] - Scores[correct_class_index] + delta);
                    }

                }

            }
            return loss;

            
        }

        // regulariztion function L2
        // to get regularization loss
        // L is a hyperparameter that controls the strength of the L2 regularization penalty
        // L L2 Regularization strength
        // regularization loss
        public static double Get_Regularization_Loss(Matrix W, double L)
        {
            double loss = 0;
            for (int row = 0; row < W.Row; row++)
            {
                for (int col = 0; col < W.Column; col++)
                {
                    loss = loss + (W[row, col] * W[row, col]);
                }
            }
            return L * loss;
        }

        // numerical gradient
        public static Matrix Get_Numerical_Gradient(Matrix input_data, Matrix Bias, Matrix W,int correct_label)
        {
            // declare the graident matrix to be returned
            double[,] Gradient_array = new double[W.Row, W.Column];
            Matrix Gradient;

            // delta
            double delta = 1;

            // get the f(x) = orginal loss
            Matrix Scores = Get_Scores(input_data, W, Bias);
            double orginal_loss = Get_SVM_Loss(Scores,correct_label,delta);
            Matrix Current_W = W;

            //  dims
            // get the f(x+h) = new loss
            double h = 0.0001;
            for (int row=0; row < W.Row; row++)
            {
                for(int col = 0; col < W.Column; col++)
                {
                    
                    Matrix W_h = Current_W;
                    W_h[row, col] = W_h[row, col] + h;
                    Matrix New_Scores = Get_Scores(input_data, W_h, Bias);
                    double new_loss = Get_SVM_Loss(New_Scores, correct_label, delta);

                    // [f(x+h) - f(x)] / h
                    double gradient = (new_loss - orginal_loss) / h;
                    
                    Gradient_array[row, col] = gradient;
                }
            }
           


            Gradient = new Matrix(Gradient_array);

            return Gradient;
            
        }


    }
    
}
