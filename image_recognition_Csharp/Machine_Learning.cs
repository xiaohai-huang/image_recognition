using System;
using System.Diagnostics;
using System.Collections.Generic;


namespace image_recognition_Csharp
{
    static class ML
    {
 
        
        public static class SVM
        {
            // score function
        /// <summary>
        /// Get the score matrix, only works for one instance
        /// </summary>
        /// <param name="input_data"></param>
        /// <param name="W"></param>
        /// <param name="Bias"></param>
        /// <returns>a matrix of scores</returns>
        public static Matrix Get_Scores(Matrix input_data, Matrix W, Matrix Bias)
        {
            // beacuse Bias is one column matrix,
            // so this method only works for one instance.
            if(input_data.Column!=1){throw new Exception("only one example each time");}
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
                double loss = Get_SVM_Loss(score,(int)Y_train[example_index],delta:1);
                total_loss=total_loss+loss;
            }
            double loss_mean = total_loss/num_of_train_examples;
            return loss_mean;
        }

        public static double Get_Full_SVM_Loss_with_Regularization(Matrix X_train, Matrix Y_train,Matrix Bias, Matrix W, double L2_strength=0.1)
        {
            double full_loss = Get_Full_SVM_Loss(X_train,Y_train,Bias,W)+Get_Regularization_Loss(W,L2_strength);
            return full_loss;
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
            double Fx = Get_Full_SVM_Loss(X_train,Y_train,Bias,W);
            for (int row=0;row<W.Row;row++)
            {
                for(int col=0;col<W.Column;col++)
                {
                    
                    
                    
                    double old_w = W[row,col];
                    W[row,col]=W[row,col]+h;

                    double Fx_h= Get_Full_SVM_Loss(X_train,Y_train,Bias,W);
                    
                    gradient[row,col]=(Fx_h-Fx)/h;

                    W[row,col]=old_w;
                }
            }
            return gradient;
        }
        /// <summary>
        /// Calculate the gradient by passing loss function
        /// </summary>
        /// <param name="loss_func">the loss function, that only takes W as the only parameter and return the loss</param>
        /// <param name="W">Weight</param>
        /// <returns>a matrix containing gradient</returns>
        public static Matrix Eval_Numerical_Gradient(Func<Matrix,double> loss_func, Matrix W)
        {
            double h = 0.00001;
            Matrix gradient=new Matrix(W.Shape);
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
        /// Calculate the accurate rate, errors/num_of_samples, using the prediction result directly
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

        /// <summary>
        /// Calculate the accurate rate, errors/num_of_samples
        /// </summary>
        /// <param name="X_test"></param>
        /// <param name="Y_test"></param>
        /// <param name="W"></param>
        /// <returns>accurate rate</returns>
        public static double Get_accuracy(Matrix X_test, Matrix Y_test, Matrix W)
        {
            Matrix prediction = Matrix.Get_Max((W*X_test));
            double accurate_rate = Get_accuracy(prediction,Y_test);
            return accurate_rate;
        }
        
        
        /// <summary>
        /// calculate numerical graident to do gradient descent, and return the W
        /// </summary>
        /// <param name="X_train">train data, must be strectch into columns</param>
        /// <param name="Y_train">the labels, one column</param>
        /// <param name="write_to_file">save the W as a text file</param>
        /// <param name="fileName">the text file's name</param>
        /// <returns>the matrix W</returns>
        public static Matrix Train_model(Matrix X_train, Matrix Y_train,double step_size=0.001,double min_loss=1,bool verbose=true, string fileName="")
        {
            // row = number of classes, column is number of pixels
            Matrix W = new Matrix(Y_train.Row,X_train.Row).Set_num(0.2);

            // bias one column
            Matrix Bias = new Matrix(Y_train.Row,1).Set_num(0.5);

            // update the W
            int time_tried=0;
            while(true)
            {
                // -----main part of updating
                Matrix grad=Eval_Numerical_Gradient(X_train,Y_train,Bias,W);
                W+=-step_size*grad;
                // ------

                // display progress
                double loss = Get_Full_SVM_Loss(X_train,Y_train,Bias,W);
                if(verbose==true)
                {
                    Console.WriteLine($"The current loss is {loss}");
                    Console.WriteLine($"Time tried: {time_tried}");
                }

                // write to file
                if(fileName!="")
                {
                    string W_text=W.Return_String()+$"\nloss: {loss}\ntime tried: {time_tried}";
                    Matrix.WriteToFile(W_text,fileName);
                }
                
                time_tried++;

                // terminate section
                if(loss<min_loss){return W;}
            }

        }
        
        /// <summary>
        /// calculate numerical graident to do gradient descent, and return the W by passing loss function
        /// </summary>
        /// <param name="loss_func">the loss function</param>
        /// <param name="W">the W to be updated</param>
        /// <param name="step_size"></param>
        /// <param name="min_loss">the loss, stop point</param>
        /// <param name="verbose">print the process message</param>
        /// <param name="fileName">save W as a text file</param>
        /// <returns>the updated W</returns>
        public static Matrix Train_model(Func<Matrix,double> loss_func,Matrix W,double step_size=0.001,double min_loss=1,bool verbose=true,string fileName="")
        {
            Matrix new_W = W;
            // update W
            int time_tried=0;
            while(true)
            {
                // ------updating part
                Matrix grad = Eval_Numerical_Gradient(loss_func,new_W);
                new_W = new_W +(-step_size*grad);
                // -------

                // display progress
                double loss = loss_func(new_W);
                if(verbose==true)
                {
                    Console.WriteLine($"The current loss is {loss}");
                    Console.WriteLine($"Time tried: {time_tried}");
                }

                // write to file
                if(fileName!="")
                {
                    string W_text=W.Return_String()+$"\nloss: {loss}\ntime tried: {time_tried}";
                    Matrix.WriteToFile(W_text,fileName);
                }
                
                time_tried++;


                // terminate section
                if(loss<min_loss){break;}
            }
            return new_W;
        }
    
        }
        
        public static class LogisticRegression
        {
            /// <summary>
            /// Sigmoid function, take Z, return number between 0 - 1
            /// </summary>
            /// <param name="Z">W.T * X + b</param>
            /// <returns>A, size is depended on the Z</returns>
            public static Matrix Sigmoid(Matrix Z)
            {
                // Matrix A = new Matrix(Z.Shape);
                // for(int row=0;row<Z.Row;row++)
                // {
                //     for(int col=0;col<Z.Column;col++)
                //     {
                //         A[row,col] = 1/(  1+Math.Pow(Math.E,-Z[row,col])  );
                //     }
                // }

                Matrix A = new Matrix(Z.Shape);
                A = 1/(1+Matrix.Exp(-1*Z));
                return A;
            }

            public static double Loss_Function(double a, double y)
            {
                return -(y*Math.Log(a) + (1-y)*Math.Log(1-a) );
            }

            public static double Cost_Function(Matrix A, Matrix Y)
            {
                double m = A.Column;
                double cost =0;
                
                for(int i=0;i<m;i++)
                {
                    double a = A[0,i];
                    double y = Y[0,i];
                    cost=cost+Loss_Function(a,y);
                }
                return cost;
            }

            /// <summary>
            /// Train the model and perform gradient descent
            /// </summary>
            /// <param name="X">each column is an example(has been stretched into columns)</param>
            /// <param name="Y">1 row matrix</param>
            /// <param name="Learning_Rate"></param>
            /// <param name="Save_Path"></param>
            /// <returns>an array contains W, and Bias</returns>
            public static object[] Train_model(Matrix X, Matrix Y,double Learning_Rate=0.003, string Save_Path="")
            {
                object[] result = new object[2];
                // initialize W, vertical vector
                Matrix W = Matrix.Random_Matrix(X.Get_Column(0).Row,1)*0.01;

                // initialize some variables
                double m = X.Column; // # of examples
                Matrix b = new Matrix(1,1);        // Bias
                Matrix Z;
                Matrix A;
                Matrix dZ;
                Matrix dW;
                Matrix dB;


                int times=0;
                while (true)
                {
                    Z = W.T * X + b;
                    A = ML.LogisticRegression.Sigmoid(Z);
                    dZ = A-Y;
                    dW = 1/m * X*dZ.T;
                    dB = 1/m * Matrix.Sum(dZ);
                    
                    // graident descent
                    W = W - Learning_Rate*dW;
                    b = b - Learning_Rate*dB;
                    
                    times++;
                    Console.WriteLine(times);
                    // terminate if A==Y
                    if(A.Is_Equal(Y)){break;}
                }
                result[0] = W;
                result[1] = b;
                if(Save_Path!="")
                {
                    W.SaveMatrix(Save_Path);
                }

                return result;
            }
        
            public static object[] Propagate(Matrix w, Matrix b, Matrix X, Matrix Y)
            {
                /*
                Arguments:
                w -- weights, a numpy array of size (num_px * num_px * 3, 1)
                b -- bias, a scalar
                X -- data of size (num_px * num_px * 3, number of examples)
                Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

                Return:
                cost -- negative log-likelihood cost for logistic regression
                dw -- gradient of the loss with respect to w, thus same shape as w
                db -- gradient of the loss with respect to b, thus same shape as b
                */
                // if b is just a number
                b = new Matrix(w.T.Row,X.Column).Set_num(b[0,0]);
                double m = X.Shape[1]; // # examples
                // FORWARD PROPAGATION (FROM X TO COST)
                Matrix A = Sigmoid(w.T*X+b);    // compute activation
                Matrix cost = -1/m * Matrix.Sum(Y.Multiply(Matrix.Log(A))+(1 - Y).Multiply( Matrix.Log(1 - A)) ); //compute cost

                // BACKWARD PROPAGATION (TO FIND GRAD)
                Matrix dz = (1/m)*(A - Y);
                Matrix dw = X * dz.T;
                Matrix db = Matrix.Sum(dz);

                Debug.Assert(dw.Size == w.Size);
                
                Dictionary<string,Matrix> grads = new Dictionary<string, Matrix>();
                grads.Add("dw",dw);
                grads.Add("db",db);
                
                object[] result = new object[2];
                result[0] = grads;
                result[1] = cost;

                return result;
            }
        }
    }
    
}
