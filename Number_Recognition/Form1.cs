using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
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
        public Form1()
        {
            InitializeComponent();
            init_data();
            //file = new Byte_File("data\\train-images.idx3-ubyte", "data\\train-labels.idx1-ubyte");
            numbers = new Machine(bit_xor_output, init_res_array, bit_xor_input, 1, normalizing_value: 1);
            numbers.add_layer(2, 2, Machine.Layer.SQAUSH_FUNC.SIGMOID);
            numbers.add_layer(2, 1, Machine.Layer.SQAUSH_FUNC.SIGMOID);
            numbers.train(2000);

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
    }
}
