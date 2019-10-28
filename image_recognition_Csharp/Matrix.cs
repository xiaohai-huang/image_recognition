using System;
using System.Collections.Generic;


namespace image_recognition_Csharp
{
    class Matrix
    {
        // class fields
        private double[,] data;
        public int Row
        {
            get { return data.GetLength(0); }
            private set { Row = value; }
        }
        public int Column
        {
            get { return data.GetLength(1); }
            private set { Column = value; }
        }
        /// <summary>
        /// return the size of the matrix as a string
        /// </summary>
        /// <value></value>
        public string Size
        {
            get { return $"{this.Row} X {this.Column}"; }
        }

        //[] overload
        /// <summary>
        /// get the element of the matrix using the given row and col index
        /// </summary>
        /// <value></value>
        public double this[int row, int col]
        {

            get
            {
                return data[row, col];
                throw new ArgumentException("Invalid index!");
            }
            set
            {
                data[row, col] = value;
            }

        }
        /// <summary>
        /// only works for 1 column matrix, get the specific value
        /// </summary>
        /// <value></value>
        public double this[int row]
        {


            get
            {
                if (this.Column == 1)
                {
                return data[row, 0];
                }
                else
                {
                    Console.WriteLine("Invalid indexing");
                    throw new ArgumentException();
                    
                }
                
            }
            set
            {
                if (this.Column == 1)
                {

                data[row, 0] = value;
                }
                else
                {
                    Console.WriteLine("Invalid indexing");
                    throw new ArgumentException();
                    
                }
            }

        }

        /// <summary>
        /// construct an empty matrix with specific row and column
        /// </summary>
        /// <param name="row"></param>
        /// <param name="col"></param>
        public Matrix(int row, int col)
        {
            data = new double[row, col];
        }
        /// <summary>
        /// construct a matrix using 2D array
        /// </summary>
        /// <param name="input">a 2D array</param>
        public Matrix(double[,] input)
        {
            data=input;
        }
        /// <summary>
        /// construct a matrix using jagged array, and concatenate them like stretch images
        /// </summary>
        /// <param name="jagged"></param>
        public Matrix(double[][] jagged)
        {
            // concatenation is low efficiency
            // //Matrix new_matrix=new Matrix(jagged[0].GetLength(0),jagged.GetLength(0));
            // Matrix new_matrix = new Matrix(row:jagged[0].GetLength(0),col:1);
            // foreach(double[] row in jagged)
            // {
            //     Matrix matrix = new Matrix(row);
            //     matrix=matrix.Reshape(1);
            //     new_matrix=new_matrix.Concatenate(matrix);
            // }
            // new_matrix=new_matrix.Remove_Column(0);

            // this.data=new_matrix.data;

            //----- 
            // construct an empty matrix to be populated
            Matrix new_matrix=new Matrix(row:jagged[0].GetLength(0),col:jagged.GetLength(0));

            //
            int col_index=0;
            foreach(double[] row in jagged)
            {
                for(int inner_row_index=0;inner_row_index<row.Length;inner_row_index++)
                {
                    new_matrix[inner_row_index,col_index]=row[inner_row_index];
                }
                col_index++;
            }
            this.data=new_matrix.data;
        }
        
        /// <summary>
        /// if input array has only 1 D,reshape it to two D array, e.g 784 => 28 * 28
        /// </summary>
        /// <param name="one_D_array">one dimension array</param>
        public Matrix(double[] one_D_array)
        {
            if (one_D_array.Length == 2)
            {
                double[,] two_D_version_ = new double[2, 1];
                two_D_version_[0, 0] = one_D_array[0];
                two_D_version_[1, 0] = one_D_array[1];
                data = two_D_version_;

            }
            else
            {
                if(Math.Sqrt(one_D_array.Length)>(int)Math.Sqrt(one_D_array.Length))
                {
                    throw new ArgumentException("The 1D array cannot be fully square rooted.");
                }

                // sqrt return a double
                // in order to reshape an array, the array has to be ** 1/2
                // not just /2 in length.
                double[,] two_D_version = new double[(int)(Math.Sqrt(one_D_array.Length)), (int)(Math.Sqrt(one_D_array.Length))];
                List<double> input_list = new List<double>();
                for (int x = 0; x < one_D_array.Length; x++)
                {
                    input_list.Add(one_D_array[x]);
                }

                int index = 0;

                for (int row = 0; row < two_D_version.GetLength(0); row++)
                {
                    for (int col = 0; col < two_D_version.GetLength(1); col++)
                    {
                        try
                        {
                            two_D_version[row, col] = input_list[index];

                        }
                        catch
                        {
                            Console.WriteLine("out of range" + index);

                        }
                        index++;
                    }
                }

                data = two_D_version;
            }
        }

        /// <summary>
        /// Construct a matrix using a text file
        /// </summary>
        /// <param name="file_path">the file contains matrix</param>
        public Matrix(string file_path)
        {
            string[] lines = System.IO.File.ReadAllLines(file_path);

            List<string> matrix_part = new List<string>();
            // get the matrix part, and remove the {},
            foreach(string line in lines)
            {
                try// if empty break
                {
                    if(line.Substring(0,1)!="{")// if not { break
                    {
                        break;
                    }

                }
                catch (ArgumentOutOfRangeException)
                {
                    break;
                }
                matrix_part.Add(line.Substring(1,line.Length-3));//-2 means skip "},"
            }  

            int col_num = matrix_part[0].Split(",").Length;
            double[,] matrix_arr=new double[matrix_part.Count,col_num];
            int row=0;
            foreach(string line in matrix_part)
            {
                string[] nums_str = line.Split(",");
                for(int col=0;col<nums_str.Length;col++)
                {
                    matrix_arr[row,col]=double.Parse(nums_str[col]);
                }
                row++;
            }
            this.data=matrix_arr;
            

            
           
            
            //this.data=matrix_arr;
        } 
    
        
        /// <summary>
        /// matrix dot product
        /// </summary>
        /// <param name="right_matrix"></param>
        /// <returns></returns>
        public Matrix Dot(Matrix right_matrix)
        {
            Matrix output_matrix;

            // check whether row == col
            if (this.Column == right_matrix.Row)//valid
            {
                output_matrix = new Matrix(row:this.Row,col:right_matrix.Column);// output matrix initialize
                
                for (int row =0; row < this.Row;row++)
                {
                    for(int col=0; col < right_matrix.Column; col++)
                    {
                        output_matrix[row, col] = Get_one_cell(row, col,right_matrix);
                    }
                }
                return output_matrix;
            }
            else// if input is invalid
            {
                Console.WriteLine("left column and right row is not the same");
                Console.WriteLine($"{this.Column} != {right_matrix.Row}");
                throw new ArgumentException();
                
            }
            
        }
        private double Get_one_cell(int l_row_index, int r_col_index, Matrix R_matrix)
        {
            double sum=0;

            // because the left column has to be the same as the right row
            // just use left row's length
            for (int x = 0; x < this.Column; x++)
            {
                sum = sum + (this[l_row_index, x] * R_matrix[x, r_col_index]);
            }

            return sum;
        }
        public static Matrix operator *(Matrix left, Matrix right)
        {
            return left.Dot(right);
        }

        /// <summary>
        /// return the sum of the whole matrix
        /// </summary>
        /// <returns></returns>
        public double Sum()
        {
            double result=0;
            for (int row = 0; row < this.Row; row++)
            {
                for(int col = 0; col < this.Column; col++)
                {
                    result = result + this[row, col];
                }
            }
            return result;
        }
        /// <summary>
        /// element-wise addition
        /// </summary>
        /// <param name="right_matrix"></param>
        /// <returns></returns>
        public Matrix Add(Matrix right_matrix)
        {
            Matrix added_matrix;
            // check the size
            if (this.Size != right_matrix.Size)
            {
                Console.WriteLine("Cannot add these two matries together");
                throw new ArgumentException($"Orginal: {this.Size} is not equal to the right matrix: {right_matrix.Size}");
                
            }
            else
            {
                added_matrix = new Matrix(this.Row, this.Column);

                for(int row = 0; row < added_matrix.Row; row++)
                {
                    for(int col =0; col < added_matrix.Column; col++)
                    {
                        added_matrix[row, col] = this[row, col] + right_matrix[row, col];
                    }
                }
            }
            return added_matrix;
            
        }
        /// <summary>
        /// element-wise addition
        /// </summary>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <returns></returns>
        public static Matrix operator +(Matrix left, Matrix right) 
        {
            Matrix result;
            result = left.Add(right);

            return result;
        }

        /// <summary>
        /// element-wise multiplication
        /// </summary>
        /// <param name="num">the number to be multiplied</param>
        /// <returns>new matrix</returns>
        public Matrix Multiply(double num)
        {
            Matrix Multiplied_matrix=new Matrix(this.Row,this.Column);

            for(int row=0; row < this.Row; row++)
            {
                for(int col=0; col < this.Column; col++)
                {
                    Multiplied_matrix[row, col] = this[row, col] * num;
                }
            }
            return Multiplied_matrix;
        }
        public static Matrix operator *(Matrix left, double right)
        {
            Matrix result;
            result = left.Multiply(right);
            return result;
        }
        public static Matrix operator *(double left, Matrix right)
        {
            Matrix result;
            result = right.Multiply(left);
            return result;
        }

        /// <summary>
        /// set all elements to a specific nummber
        /// </summary>
        /// <param name="num">the number to set</param>
        /// <returns>a matrix which is full of the number</returns>
        public Matrix Set_num(double num)
        {
            Matrix new_matrix = new Matrix(this.Row,this.Column);
            for (int row = 0; row < new_matrix.Row; row++)
            {
                for(int col = 0; col < new_matrix.Column; col++)
                {
                    new_matrix[row, col] = num;
                }
            }
            return new_matrix;
        }


        

        
        /// <summary>
        /// reshape the matrix,supply row and column
        /// </summary>
        /// <param name="row">the number of rows</param>
        /// <param name="col">the number of columns</param>
        /// <returns>return a row x col matrix</returns>   
        public Matrix Reshape(int row, int col)
        {
            if (row * col != this.Row * this.Column)
            {
                Console.WriteLine("cannot reshpe this matrix");
                Console.WriteLine($"Orginal: {this.Row} X {this.Column} ===> Output: {row} X {col}");
                return this;
            }
            // create a list to store original matrix values (each cell)
            List<double> original_values= new List<double>();
            for (int original_row =0; original_row < this.Row; original_row++)
            {
                for(int orginal_col =0;orginal_col<this.Column; orginal_col++)
                {
                    original_values.Add(this[original_row, orginal_col]);
                }
            }

            Matrix shapped_matrix;
            shapped_matrix = new Matrix(row, col);
            int list_index = 0;

            for (int shapped_row =0; shapped_row < shapped_matrix.Row; shapped_row++)
            {
                for(int shapped_col = 0; shapped_col < shapped_matrix.Column; shapped_col++)
                {
                    shapped_matrix[shapped_row, shapped_col] = original_values[list_index];
                    list_index++;
                }
            }
            return shapped_matrix;

            
        }

        /// <summary>
        /// Reshapre the matrix using column number only
        /// </summary>
        /// <param name="col">the number of columns</param>
        /// <returns>a reshaped matrix</returns>
        public Matrix Reshape(int col)
        {
            int row = (this.Row * this.Column) / col;
            return this.Reshape(row, col);

            
        }

        
        public override string ToString()
        {
            string result;
           
            result = $"Row X Column : {this.Row} X {this.Column}";

            return result;
        }
        /// <summary>
        /// Display the matrix
        /// </summary>
        /// <param name="decimal_num">the number of decimal spaces to use</param>
        public void Display(int decimal_num=2)
        {
            for(int row = 0; row < this.Row; row++)
            {
                for(int col = 0; col < this.Column; col++)
                {
                    Console.Write(Math.Round(this[row, col],decimal_num)+"  ");
                }
                Console.WriteLine();
            }
            Console.WriteLine($"\nThis is a {this.Row} x {this.Column} Matrix");
        }

        /// <summary>
        /// turn the whole matrix into a string
        /// </summary>
        /// <returns>a string containing the whole matrix</returns>
        public string Return_String()
        {
            string text = "";
            for (int row = 0; row < this.Row; row++)
            {
                text=text+"{";
                for (int col = 0; col < this.Column; col++)
                {
                    if(col==this.Column-1)
                    {
                        text=text+(this[row, col]);
                    }
                    else
                    {
                        text=text+(this[row, col] + ",");
                    }
                }
                text = text + "},\n";
            }
            return text;
        }

        
        /// <summary>
        /// turn the specific column into a new matrix (one column)
        /// </summary>
        /// <param name="col_index">the specific column</param>
        /// <returns>one column matrix</returns>
        public Matrix Get_Column(int col_index)
        {
            Matrix new_matrix=new Matrix(this.Row,1);
            for (int row=0;row<this.Row;row++)
            {
                new_matrix[row,0]=this[row,col_index];
            }
            return new_matrix;
        }
        /// <summary>
        /// Concatenate two matries together
        /// </summary>
        /// <param name="right">the matrix to be combined</param>
        /// <returns>return the combined matrix</returns>
        public Matrix Concatenate(Matrix right)
        {
            // check row number
            if(this.Row!=right.Row)
            {
                throw new ArgumentException($"{this.Row}!={right.Row}\n row number has to be the same");
            }
            Matrix new_matrix = new  Matrix(this.Row,this.Column+right.Column);
            // populate the new matrix by using the left matrix
            for(int row=0;row<this.Row;row++)
            {
                for(int col=0;col<this.Column;col++)
                {
                    new_matrix[row,col]=this[row,col];
                }
            }

            // using the right matrix
            for(int row=0;row<right.Row;row++)
            {
                for(int col=this.Column;col<this.Column+right.Column;col++)
                {
                    new_matrix[row,col]=right[row,col-this.Column];
                }
            }
            return new_matrix;

        }
        /// <summary>
        /// Remove a specific column in the matrix
        /// </summary>
        /// <param name="col_index">the column's index to be removed</param>
        /// <returns>return a new matrix after removing</returns>
        public Matrix Remove_Column(int col_index)
        {
            Matrix new_matrix;
            if(col_index!=0)
            {
                new_matrix=this.Get_Column(0);
                
                // initialize with one column, therefore, col should start with 1
                for (int col=1;col<this.Column;col++)
                {
                    if(col!=col_index)
                    {
                        new_matrix=new_matrix.Concatenate(this.Get_Column(col));
                    }
                }
            }
            else
            {
                // initialize with one column, therefore, col should start with 2
                new_matrix=this.Get_Column(1);
                for (int col=2;col<this.Column;col++)
                {
                    
                    new_matrix=new_matrix.Concatenate(this.Get_Column(col));
                    
                }
            }
            return new_matrix;
            
                
            
        }

        private static double Get_one_col_max(Matrix matrix,int col_index)
        {
            double max=0;
            int max_index=0;
            for(int row=0;row<matrix.Row;row++)
            {
                
                if(matrix[row,col_index]>max)
                {
                    max=matrix[row,col_index];
                    max_index=row;
                }
            
            }
            return max_index;
        }

         
         /// <summary>
         /// find the index with max score in each column
         /// </summary>
         /// <param name="matrix">The matrix to be searched</param>
         /// <returns>Return an one column matrix</returns>
        public static Matrix Get_Max(Matrix matrix)
        {
            // the number of rows dependes on the column number 
            // of the parameter
            Matrix max_matrix = new Matrix(row:matrix.Column,col:1);
            
            List<double> max_list = new List<double>(); 
            for(int column_index=0;column_index<matrix.Column;column_index++)
            {
                max_list.Add(Get_one_col_max(matrix,column_index));
            }
            
            for(int row=0;row<max_list.Count;row++)
            {
                max_matrix[row]=max_list[row];
            }
            return max_matrix;
        }
        /// <summary>
        /// sample data for jagged array
        /// </summary>
        /// <param name="data">jagged array</param>
        /// <param name="num_of_examples">number of examples</param>
        /// <returns>sample data jagged array</returns>
        public static double[][] sample_training_data(double[][] data, int num_of_examples)
        {
            if(data.GetLength(0)<num_of_examples)
            {
                throw new ArgumentException($"the size of data are samaller enough, cannot be sampled further");
            }else if(data.GetLength(0)==num_of_examples)
            {
                return data;
            }
            
            double[][] result = new double[num_of_examples][];
            int num=0;
            foreach(double[] row in data)
            {
                if(num==num_of_examples){break;}
                result[num]=row;
                num++;
            }
            return result;

        }
        /// <summary>
        /// sample data for 1D array
        /// </summary>
        /// <param name="data">1D array</param>
        /// <param name="num_of_examples"></param>
        /// <returns>sample data 1D array</returns>
        public static double[] sample_training_data(double[] data, int num_of_examples)
        {
            if(data.GetLength(0)<num_of_examples)
            {
                throw new ArgumentException($"the size of data are samaller enough, cannot be sampled further");
            }
            else if(data.GetLength(0)==num_of_examples)
            {
                return data;
            }

            double[] result= new List<double>(data).GetRange(0,num_of_examples).ToArray();
            return result;

        }
        public static void WriteToFile(string text_to_write,string file_path)
        {
            System.IO.File.WriteAllText(file_path,text_to_write);
        }
        
    
    }
}
