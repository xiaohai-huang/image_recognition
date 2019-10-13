using System;
using System.Collections.Generic;
using System.Text;

namespace image_recognition_Csharp
{
    class Matrix
    {
        double[,] data;
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
                return data[row,col];
                throw new ArgumentException("Invalid index!");
            }
            set
            {
                data[row, col] = value;
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
            else
            {
                Console.WriteLine("left column and right row is not the same");
                Console.WriteLine($"{this.Column} != {right_matrix.Row}");
                return right_matrix;// if input is invalid
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
    }
}
