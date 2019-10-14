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
        public static double Get_Regularization_Loss(Matrix W,double L)
        {
            double loss = 0;
            for(int row = 0; row < W.Row; row++)
            {
                for(int col = 0; col < W.Column; col++)
                {
                    loss = loss + (W[row, col]* W[row, col]);
                }
            }
            return L*loss;
        }



    }
    
}
