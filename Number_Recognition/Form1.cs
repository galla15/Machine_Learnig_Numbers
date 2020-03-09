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
        public Form1()
        {
            InitializeComponent();
            file = new Byte_File("data\\train-images.idx3-ubyte", "data\\train-labels.idx1-ubyte");
            numbers = new Machine(file.labelSet, file.dataset);
            numbers.add_layer(784, 16);
            numbers.add_layer(16, 10);
            numbers.train(1);

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
