using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace Number_Recognition
{
    class Byte_File
    {
        private string image_path;
        private string label_path;
        private uint number_of_labels;
        private uint number_of_image;
        private uint number_of_rows;
        private uint number_of_columns;
        public byte[][] dataset { get; private set; }
        public byte[] labelSet { get; private set; }
        private FileStream file;

        public Byte_File(string image_path, string label_path)
        {
            this.image_path = image_path;
            this.label_path = label_path;
            read_mnist_images();
            read_mnist_labels();
        }

        private int get_parameters()
        {
            string param = string.Empty;

            for (int i = 0; i < 4; i++)
            {
                byte buf = (byte)file.ReadByte();
                if (buf < 10) param += "0" + buf.ToString();
                else param += Convert.ToString(buf, 16);
            }

     
            return int.Parse(param, System.Globalization.NumberStyles.HexNumber);
        }
        private void read_mnist_images()
        {
            file = new FileStream(image_path, FileMode.Open, FileAccess.Read);
            
            //Checking if it's a mnist file
            if(get_parameters() != 2051)
            {
                throw new Exception("File is not of type mnist!");
            }

            number_of_image = (uint)get_parameters();
            number_of_rows = (uint)get_parameters();
            number_of_columns = (uint)get_parameters();

            dataset = new byte[number_of_image][];

            for(uint i = 0; i < number_of_image; i++)
            {
                dataset[i] = new byte[number_of_rows * number_of_columns];

                for(int j = 0; j < dataset[i].Length; j++)
                {
                    dataset[i][j] = (byte)file.ReadByte();
                }
            }

            file.Close();
        }

        private void read_mnist_labels()
        {
            file = new FileStream(label_path, FileMode.Open, FileAccess.Read);

            //Checking if it's a mnist file
            if (get_parameters() != 2049)
            {
                throw new Exception("File is not of type mnist!");
            }

            number_of_labels = (uint)get_parameters();

            if(number_of_labels != number_of_image)
            {
                throw new Exception("Wrong label file!");
            }

            labelSet = new byte[number_of_labels];

            for (uint i = 0; i < number_of_labels; i++)
            {
                labelSet[i] = (byte)file.ReadByte();
            }

            file.Close();
        }

        public Emgu.CV.Image<Emgu.CV.Structure.Gray, byte> printImage(int index)
        {
            Emgu.CV.Image < Emgu.CV.Structure.Gray, byte> image = new Emgu.CV.Image<Emgu.CV.Structure.Gray, byte>((int)number_of_columns, (int)number_of_rows);
            image.Bytes = dataset[index];

            Emgu.CV.CvInvoke.Resize(image, image, new System.Drawing.Size(200, 200));

            return image;
        }

        public uint get_image_value(int index)
        {
            return labelSet[index];
        }

    }
}
