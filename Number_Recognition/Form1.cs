using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Number_Recognition
{
    public partial class Form1 : Form
    {
        Byte_File file;
        Machine numbers;

        byte[][] bit_xor_input = new byte[4][];
        byte[] bit_xor_output = new byte[] { 0, 1, 1, 0 };

        public void init_data()
        {
            bit_xor_input[0] = new byte[2] { 0, 0 };
            bit_xor_input[1] = new byte[2] { 0, 1 };
            bit_xor_input[2] = new byte[2] { 1, 0 };
            bit_xor_input[3] = new byte[2] { 1, 1 };
        }

        public double[][] init_res_array(byte[] arr)
        {
            double[][] res = new double[arr.Length][];

            for(int i = 0; i < arr.Length; i++)
            {
                res[i] = new double[1];
                res[i][0] = (double)arr[i];
            }

            return res; 
        }

        byte[][] input = new byte[2][];
        
        byte[] output = new byte[] { 1, 2 };

        public void init()
        {
            input[0] = new byte[] { 0 };
            input[1] = new byte[] { 1 };
        }


        public double[][] init_labels_array(byte[] arr)
        {
            double[][] output = new double[arr.Length][];
            
            for(int i = 0; i < arr.Length; i++)
            {
                output[i] = new double[10];

                Array.Clear(output[i], 0, output[i].Length);

                output[i][arr[i]] = 1;
            }

            return output;
        }


        public Form1()
        {
            InitializeComponent();
            file = new Byte_File("data\\train-images.idx3-ubyte", "data\\train-labels.idx1-ubyte");

            /*numbers = new Machine(file.labelSet, file.dataset, 10, init_result_array: init_labels_array, learning_rate:0.01);
            numbers.add_layer(file.dataset[0].Length, 16, func: Machine.Squash_func.SQUASH_FUNC.RELU);
            numbers.add_layer(16, 16, Machine.Squash_func.SQUASH_FUNC.RELU);
            numbers.add_layer(16, 10, Machine.Squash_func.SQUASH_FUNC.RELU);
            numbers.train(10);*/

            StocasticDualCoordianteAscent model = new StocasticDualCoordianteAscent(file.get_data_set(), file.get_labels());


            /*init_data();
            numbers = new Machine(bit_xor_output, bit_xor_input, 1, init_result_array: init_res_array, normalizing_value: 1, learning_rate:0.1);
            numbers.add_layer(2, 2, Machine.Squash_func.SQUASH_FUNC.SIGMOID);
            numbers.add_layer(2, 1, Machine.Squash_func.SQUASH_FUNC.SIGMOID);
            numbers.train(1000000);*/

            //init();
            //numbers = new Machine(output, input, 1, init_result_array: init_res_array, normalizing_value: 1);


        }

        private void button1_Click(object sender, EventArgs e)
        {
            int index = 0;
            try
            {
                index = Convert.ToInt32(textBox1.Text);
            }
            catch (FormatException)
            {
                return;
            }

            imageBox1.Image = file.printImage(index);
            textBox2.Text = file.get_image_value(index).ToString();
        }

        private void button2_Click(object sender, EventArgs e)
        {
            double[] data = file.getImage(Convert.ToInt32(textBox1.Text));

            Vector<double> res = numbers.run(data);

            Console.WriteLine("Image value is : \n" + res.ToString());

        }
    }
}
