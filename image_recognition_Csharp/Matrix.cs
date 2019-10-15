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
        public string Size
        {
            get { return $"{this.Row} X {this.Column}"; }
        }

        //[] overload
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

        // constructor
        public Matrix(int row, int col)
        {
            data = new double[row, col];
        }

        public Matrix(double[,] input)
        {
            data=input;
        }
 
        // if input array has only 1 D
        // reshape it to two D array, e.g 784 => 28 * 28
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

        // jagged array
        
        // dot product
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

        // sum matrix
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
        // addition
        public Matrix Add(Matrix right_matrix)
        {
            Matrix added_matrix;
            // check the size
            if (this.Size != right_matrix.Size)
            {
                Console.WriteLine("Cannot add these two matries together");
                Console.WriteLine($"Orginal: {this.Size} is not equal to Output: {right_matrix.Size}");
                return this;
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
        public static Matrix operator +(Matrix left, Matrix right) 
        {
            Matrix result;
            result = left.Add(right);

            return result;
        }

        // multiplication
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

        // set number
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

        // get the maxinum number of the whole matrix
        public double Get_Max()
        {
            Matrix one_D = this.Reshape(1);
            double max_num=one_D[0];
            for (int index = 0; index < one_D.Row; index++)
            {
                max_num = Math.Max(max_num, one_D[index]);
            }

            return max_num;

        }
        public int Get_Max_index()
        {
            Matrix one_D = this.Reshape(1);
            double max_num = one_D[0];
            int result = 0;
            for (int index = 0; index < one_D.Row; index++)
            {
                max_num = Math.Max(max_num, one_D[index]);
                result = index;
            }

            return result;

        }

        // reshape the matrix
        // supply row and column   
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

        // supply column only
        public Matrix Reshape(int col)
        {
            int row = (this.Row * this.Column) / col;
            return this.Reshape(row, col);

            
        }

        // display matrix
        public override string ToString()
        {
            string result;
           
            result = $"Row X Column : {this.Row} X {this.Column}";

            return result;
        }
        public void Display()
        {
            for(int row = 0; row < this.Row; row++)
            {
                for(int col = 0; col < this.Column; col++)
                {
                    Console.Write(Math.Round(this[row, col],2)+"  ");
                }
                Console.WriteLine();
            }
        }

        public string Return_String()
        {
            string text = "";
            for (int row = 0; row < this.Row; row++)
            {
                for (int col = 0; col < this.Column; col++)
                {
                   text=text+(Math.Round(this[row, col], 6) + "\t");
                }
                text = text + "\n";
            }
            return text;
        }
    }
}
