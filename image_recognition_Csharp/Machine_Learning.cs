using System;


namespace image_recognition_Csharp
{
    static class ML
    {
 

        // score function
        /// <summary>
        /// Get the score matrix
        /// </summary>
        /// <param name="input_data"></param>
        /// <param name="W"></param>
        /// <param name="Bias"></param>
        /// <returns>a matrix of scores</returns>
        public static Matrix Get_Scores(Matrix input_data, Matrix W, Matrix Bias)
        {
            Matrix Scores = W * input_data + Bias;

            return Scores;

        }

        // SVM loss function calculate the loss of the given image
        // data loss
        /// <summary>
        /// calculate the SVM loss of the given image
        /// </summary>
        /// <param name="Scores">the score matrix</param>
        /// <param name="correct_class_index">the correct index</param>
        /// <param name="delta">safe margin</param>
        /// <returns>the single loss</returns>
        private static double Get_SVM_Loss(Matrix Scores, int correct_class_index, double delta)
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


        // X_train is the data where each column is an example (e.g. 3703 x 50,000)
        // Y_train are the labels (5000 * 1)
        /// <summary>
        /// Get the SVM loss
        /// </summary>
        /// <param name="X_train">A matrix that already stretch into columns</param>
        /// <param name="Y_train">1 column matrix</param>
        /// <param name="Bias"></param>
        /// <param name="W">row number = number of classes</param>
        /// <returns>SVM loss</returns>
        public static double Get_Full_SVM_Loss(Matrix X_train, Matrix Y_train,Matrix Bias, Matrix W)
        {
            if(X_train.Column!=Y_train.Row)
            {
                throw new ArgumentException("data and labels are not at the same amount");
            }
            double total_loss=0;
            double num_of_train_examples = X_train.Column;
            for(int example_index=0;example_index<num_of_train_examples;example_index++)
            {
                Matrix example = X_train.Get_Column(example_index);
                Matrix score = Get_Scores(example,W,Bias);
                double loss = ML.Get_SVM_Loss(score,(int)Y_train[example_index],delta:1);
                total_loss=total_loss+loss;
            }
            double loss_mean = total_loss/num_of_train_examples;
            return loss_mean;
        }

        
        /// <summary>
        /// regulariztion function L2, to get regularization loss
        /// </summary>
        /// <param name="W"></param>
        /// <param name="L">L is a hyperparameter that controls the strength of the L2 regularization penalty</param>
        /// <returns>regularization loss</returns>
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

        public static Matrix Eval_Numerical_Gradient(Matrix X_train, Matrix Y_train,Matrix Bias,Matrix W)
        {
            double h = 0.00001;
            Matrix gradient=new Matrix(W.Row,W.Column);
            gradient = gradient.Set_num(0);
            double Fx = ML.Get_Full_SVM_Loss(X_train,Y_train,Bias,W);
            for (int row=0;row<W.Row;row++)
            {
                for(int col=0;col<W.Column;col++)
                {
                    
                    
                    
                    double old_w = W[row,col];
                    W[row,col]=W[row,col]+h;

                    double Fx_h= ML.Get_Full_SVM_Loss(X_train,Y_train,Bias,W);
                    
                    gradient[row,col]=(Fx_h-Fx)/h;

                    W[row,col]=old_w;
                }
            }
            return gradient;
        }
        /// <summary>
        /// Calculate the gradient by passing loss function
        /// </summary>
        /// <param name="loss_func">the loss function</param>
        /// <param name="W">Weight</param>
        /// <returns>a matrix containing gradient</returns>
        public static Matrix Eval_Numerical_Gradient(Func<Matrix,double> loss_func, Matrix W)
        {
            double h = 0.00001;
            Matrix gradient=new Matrix(W.Row,W.Column);
            gradient = gradient.Set_num(0);

            double Fx = loss_func(W);

             for (int row=0;row<W.Row;row++)
            {
                for(int col=0;col<W.Column;col++)
                {
                    
                    double old_w = W[row,col];
                    W[row,col]=W[row,col]+h;

                    double Fx_h= loss_func(W);
                    
                    gradient[row,col]=(Fx_h-Fx)/h;

                    W[row,col]=old_w;
                }
            }
            return gradient;
        }





        // <summary>
        /// Calculate the accurate rate, errors/num_of_samples
        /// </summary>
        /// <param name="Prediction">1 column matrix</param>
        /// <param name="Answer">1 column matrix</param>
        /// <returns>accurate rate</returns>
        public static double Get_accuracy(Matrix Prediction, Matrix Answer)
        {
            // check argument
            if(Prediction.Row!=Answer.Row)
            {
                throw new ArgumentException("answer's amount and prediction's amount doesn't match!");
            }else if(Prediction.Column!=1)
            {
                throw new ArgumentException("only one column argument is acceptable");
            }
            double num_of_samples = Prediction.Row;
            double errors=0;
            for(int row_index=0;row_index<Prediction.Row;row_index++)
            {
                if(Prediction[row_index,0]==Answer[row_index,0])
                {
                    errors=errors+1;
                }
            }
            double accurate_rate = errors/num_of_samples;

            return accurate_rate;
        
    
        }
    
    
    
    }
    
}
