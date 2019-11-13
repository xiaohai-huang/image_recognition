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
            /// <returns>A, size is the same as input</returns>
            public static Matrix Sigmoid(Matrix Z)
            {
                Matrix A = new Matrix(Z.Shape);
                A = 1/(1+Matrix.Exp(-1*Z));
                return A;
            }

            /// <summary>
            /// Implement the cost function and its gradient for the propagation.[0]grads-{dict}-"dw","db". [1]cost-{matrix}
            /// </summary>
            /// <param name="w">weights, shape (Nx,1)</param>
            /// <param name="b">bias, a scalar, shape(1, 1)</param>
            /// <param name="X">data, shape(Nx, number of examples)</param>
            /// <param name="Y">"label" vector, shape (1, number of examples)</param>
            /// <returns>1D array,[0]grads-{dict}-"dw","db". [1]cost-{matrix}</returns>
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
                

                double m = X.Shape[1]; // # examples
                // FORWARD PROPAGATION (FROM X TO COST)
                Matrix Z = w.T*X+b;
                Matrix A = Sigmoid(Z);    // compute activation
                Matrix cost = (-1/m) * Matrix.Sum(Y.Multiply(Matrix.Log(A)) + (1 - Y).Multiply( Matrix.Log(1 - A)) ); //compute cost

                // BACKWARD PROPAGATION (TO FIND GRAD)
                Matrix dz = (A - Y);
                Matrix dw = (1/m) * (X * dz.T);
                Matrix db = (1/m) * (Matrix.Sum(dz));

                Debug.Assert(dw.Size == w.Size);
                
                Dictionary<string,Matrix> grads = new Dictionary<string, Matrix>();
                grads.Add("dw",dw);
                grads.Add("db",db);
                
                object[] result = new object[2];
                result[0] = grads;
                result[1] = cost;

                return result;
            }

            /// <summary>
            /// This function optimizes w and b by running a gradient descent algorithm,[0]params-{dict} containing w and b. [1]grads-{matrix}. [2]costs-{list}
            /// </summary>
            /// <param name="w">weights, 1 column Matrix</param>
            /// <param name="b">bias, a scalar</param>
            /// <param name="X">data, shape(Nx, number of examples)</param>
            /// <param name="Y">"label" vector, shape(1, number of examples)</param>
            /// <param name="num_iterations">number of iterations of the optimization loop</param>
            /// <param name="learning_rate">learning rate of the gradient descent update rule</param>
            /// <param name="print_cost">True to print the loss </param>
            /// <returns>[0]params-{dict} containing w and b. [1]grads-{dict}. [2]costs-{list}</returns>
            public static object[] Optimize(
                Matrix w, Matrix b,Matrix X, Matrix Y,
                int num_iterations,double learning_rate,
                bool print_cost= false)
            {
                /*
                Returns:
                params -- dictionary containing the weights w and bias b
                grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
                costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
                */

                // initialzie variables
                List<double> costs = new List<double>();
                Matrix db=new Matrix(1,1);
                Matrix dw=new Matrix(1,1);
                Dictionary<string,Matrix> grads= new Dictionary<string, Matrix>();

                for(int i =0; i<num_iterations;i++)
                {
                    // Cost and gradient calculation
                    object[] propagation_results = Propagate(w,b,X,Y);
                    Matrix cost = (Matrix)propagation_results[1];
                    grads = (Dictionary<string,Matrix>)propagation_results[0];
                    
                    // Retrieve derivatives from grads
                    dw = grads["dw"];
                    db = grads["db"];

                    // Update rule
                    w = w - (learning_rate*dw);
                    b = b - (learning_rate*db);

                    // Record the cost
                    costs.Add(cost[0]);

                    // Print the cost
                    if(print_cost==true)
                    {
                        Console.WriteLine($"Cost after iteration {i}: {cost[0]} ");
                    }
                }

                Dictionary<string,Matrix> Params = new Dictionary<string,Matrix>();
                Params["w"]=w;
                Params["b"]=b;

                grads["dw"]=dw;
                grads["db"]=db;

                object[] result = new object[3];
                result[0] = Params;
                result[1] = grads;
                result[2] = costs;
                return result;
            }

            /// <summary>
            /// Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b). return(1,m) containing all predictions
            /// </summary>
            /// <param name="w">weights, 1 column Matrix</param>
            /// <param name="b">bias, a scalar</param>
            /// <param name="X">data, shape(Nx, number of examples)</param>
            /// <returns>(1 row vector) containing all predictions (0/1) for the examples in X</returns>
            public static Matrix Predict(Matrix w, Matrix b, Matrix X)
            {
                
                // Compute vector "A" predicting the probabilities of a cat being present in the picture
                Matrix A = Sigmoid(w.T*X+b);
                
                // Convert the entries of a into 0 (if activation <= 0.5) or 1 (if activation > 0.5),
                // stores horizontally the predictions in a vector (1, m) Y_prediction. 
                Matrix Y_prediction = new Matrix(A.Shape);
                for(int i =0; i<A.Column;i++)
                {
                    if(A[0,i]<=0.5)
                    {
                        Y_prediction[0,i] = 0;
                    }
                    else
                    {
                        Y_prediction[0,i] = 1;
                    }
                }

                return Y_prediction;

            }

            /// <summary>
            /// Final model
            /// </summary>
            /// <param name="X_train">data, shape(Nx, number of examples)</param>
            /// <param name="Y_train">"label" vector, shape(1, number of examples)</param>
            /// <param name="X_test">data, shape(Nx, number of examples)</param>
            /// <param name="Y_test">"label" vector, shape(1, number of examples)</param>
            /// <param name="num_iterations">number of iterations of the optimization loop</param>
            /// <param name="learning_rate">learning rate of the gradient descent update rule</param>
            /// <param name="print_cost">True to print the loss</param>
            /// <returns>A dictionary containing w,b,costs,(and the input parameters)</returns>
            public static Dictionary<string,object> Model(
                Matrix X_train, Matrix Y_train,Matrix X_test, Matrix Y_test,
                int num_iterations = 2000,double learning_rate = 0.5,
                bool print_cost = false)
            {
                //  initialize parameters with zeros
                // w shape [nx,1]
                Matrix w = new Matrix(X_train.Shape[0],1);

                // b is a scalar, a number only
                Matrix b = new Matrix(1,1); 

                // Gradient descent
                object[] gradient_results = ML.LogisticRegression.Optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost);
                Dictionary<string,Matrix> parameters = (Dictionary<string,Matrix>)gradient_results[0];
                Dictionary<string,Matrix> grads = (Dictionary<string,Matrix>)gradient_results[1];
                List<double> costs = (List<double>)gradient_results[2];

                // Retrieve parameters w and b from dictionary "parameters"
                w = parameters["w"];
                b = parameters["b"];

                // Predict test/train set examples
                Matrix Y_Prediction_test = ML.LogisticRegression.Predict(w,b,X_test);
                Matrix Y_Prediction_train = ML.LogisticRegression.Predict(w,b,X_train);

                // Print train/test Errors
                // (100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100)
                Console.WriteLine("train accuracy: {0} %",(100 - Matrix.Mean(Matrix.Abs(Y_Prediction_train - Y_train)) * 100));
                Console.WriteLine("test accuracy: {0} %",(100 - Matrix.Mean(Matrix.Abs(Y_Prediction_test - Y_test)) * 100));

                Dictionary<string,object> d = new Dictionary<string, object>();
                d.Add("costs",costs);
                d.Add("Y_Prediction_test",Y_Prediction_test);
                d.Add("Y_Prediction_train",Y_Prediction_train);
                d.Add("w",w);
                d.Add("b",b);
                d.Add("learning_rate",learning_rate);
                d.Add("num_iterations",num_iterations);
                
                return d;
            }
        }
    
        public static class NN
        {
            // Defining the neural network structure
            /// <summary>
            ///  Defining the neural network structure.
            ///  Use shapes of X and Y to find n_x and n_y. 
            ///  Also, hard code the hidden layer size to be 4.
            /// </summary>
            /// <param name="X">input dataset of shape (input size, number of examples)</param>
            /// <param name="Y">labels of shape (output size, number of examples)</param>
            public static int[] Layer_sizes(Matrix X, Matrix Y)
            {
                /*
                    Returns:
                    n_x -- the size of the input layer
                    n_h -- the size of the hidden layer
                    n_y -- the size of the output layer
                */
                int n_x = X.Shape[0]; //the size of the input layer
                int n_h = 4;
                int n_y = Y.Shape[0]; //the size of the output layer
                
                int[] result = {n_x,n_h,n_y};
                
                return result;
            }

            /// <summary>
            /// Initialize the model's parameters
            /// </summary>
            /// <param name="n_x">size of the input layer</param>
            /// <param name="n_h">size of the hidden layer</param>
            /// <param name="n_y">size of the output layer</param>
            /// <returns>dictionary containing parameters:</returns>
            public static Dictionary<string,Matrix> Initialize_parameters(int n_x,int n_h, int n_y)
            {
                /*
                    Dictionary containing these parameters
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
                */
                Matrix W1 = Matrix.Random_Matrix(n_h,n_x)*0.01;
                Matrix b1 = new Matrix(n_h,1).Set_num(0);
                Matrix W2 = Matrix.Random_Matrix(n_y,n_h)*0.01;
                Matrix b2 = new Matrix(n_y,1).Set_num(0);

                Dictionary<string,Matrix> parameters = new Dictionary<string, Matrix>();
                parameters.Add("W1",W1);
                parameters.Add("b1",b1);
                parameters.Add("W2",W2);
                parameters.Add("b2",b2);

                return parameters;
            }

            /// <summary>
            /// Compute the output
            /// </summary>
            /// <param name="X">input data of size (n_x,m)</param>
            /// <param name="parameters">output of initialization function, [W1,b1,W2,b2]</param>
            /// <returns>A2(Matrix) -- The sigmoid output of the second activation cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"</returns>
            public static object[] Forward_propagation(Matrix X, Dictionary<string,Matrix> parameters)
            {
                // Retrieve each parameter from the dictionary "parameters"
                Matrix W1 = parameters["W1"];
                Matrix b1 = parameters["b1"];
                Matrix W2 = parameters["W2"];
                Matrix b2 = parameters["b2"];

                // Implement Forward Propagation to calculate A2 (probabilities)
                Matrix Z1 = W1*X+b1;
                Matrix A1 = Matrix.tanh(Z1); // tanh

                Matrix Z2 = W2*A1+b2;
                Matrix A2 = LogisticRegression.Sigmoid(Z2);

                Dictionary<string, Matrix> cache = new Dictionary<string, Matrix>();
                cache.Add("Z1",Z1);
                cache.Add("A1",A1);
                cache.Add("Z2",Z2);
                cache.Add("A2",A2);

                object[] results = {A2,cache};
                return results;
            }
            
            /// <summary>
            /// compute the value of the cost J
            /// </summary>
            /// <param name="A2">The sigmoid output of the second activation, of shape (1, number of examples)</param>
            /// <param name="Y">"true" labels vector of shape (1, number of examples)</param>
            /// <param name="parameters">dictionary containing your parameters W1, b1, W2 and b2</param>
            /// <returns></returns>
            public static double Compute_cost(Matrix A2, Matrix Y, Dictionary<string,Matrix> parameters)
            {
                int m = Y.Shape[0]; // #examples

                // Retrieve W1 and W2 from parameters
                Matrix W1 = parameters["W1"];
                Matrix W2 = parameters["W2"];

                // Compute the cost
                Matrix logprobs = Matrix.Log(A2).Multiply(Y) + (1-Y).Multiply(Matrix.Log(1-Y));
                double cost = ((1/m) * Matrix.Sum(logprobs))[0];

                return cost;
            }
            
            /// <summary>
            /// Backward propagation to compute dW1, dW2, db1, db2
            /// </summary>
            /// <param name="parameters">W1,W2,b1,b2</param>
            /// <param name="cache">Z1,A1,Z2,A2</param>
            /// <param name="X">input data (n_x,m)</param>
            /// <param name="Y">true labels vector (1, m)</param>
            /// <returns>dictionary containing gradients with respect to different parameters</returns>
            public static Dictionary<string,Matrix> Backward_propagation(Dictionary<string,Matrix> parameters,Dictionary<string,Matrix> cache,Matrix X, Matrix Y)
            {
                int m = X.Shape[1]; // #examples

                // First, retrieve W1 and W2 from the dictionary "parameters".
                Matrix W1 = parameters["W1"];
                Matrix W2 = parameters["W2"];

                // Retrieve also A1 and A2 from dictionary "cache".
                Matrix A1 = cache["A1"];
                Matrix A2 = cache["A2"];

                // Backward propagation: calculate dW1, db1, dW2, db2.
                Matrix dz2 = A2 - Y;
                Matrix dW2 = (1/m) * dz2*A1.T;
                Matrix db2 = (1/m) * Matrix.Sum(dz2); // implement axis =1. aim [n_2,1]
                
                Matrix dz1 = (W2.T*dz2).Multiply(1 - Matrix.Power(A1,2));
                Matrix dW1 = (1/m) * dz1*X.T;
                Matrix db1 = (1/m) * Matrix.Sum(dz1);

                Dictionary<string,Matrix> grads = new Dictionary<string, Matrix>();
                grads.Add("dW1",dW1);
                grads.Add("db1",db1);
                grads.Add("dW2",dW2);
                grads.Add("db2",db2);

                return grads;

            }
        }
    }
    
}
