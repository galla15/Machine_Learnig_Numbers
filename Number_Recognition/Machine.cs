using System;
using System.Collections.Generic;

namespace Number_Recognition
{
    class Machine
    {
        private byte[][] output_data;
        private byte[][] input_data;

        List<Layer> neural = new List<Layer>();
        public Machine(byte[] labels, byte[][] values)
        {
            input_data = values;
            output_data = new byte[labels.Length][];

            //Initialising the output array
            for (uint i = 0; i < labels.Length; i++)
            {
                output_data[i] = new byte[10];
                Array.Clear(output_data[i], 0, output_data[i].Length);

                output_data[i][labels[i]] = 1;
            }  
        }

        public void train(uint iterations)
        {
            for(uint i = 0; i < iterations; i++ )
            {
                double[] iter_res = new double[input_data.Length];

                for (uint j = 0; j < input_data.Length; j++)
                {
                    double[][] l_out = new double[neural.Count + 1][];
                    
                    l_out[0] = new double[input_data[j].Length];
                    Array.Copy(input_data[j], 0, l_out[0], 0, input_data[j].Length);

                    for (uint k = 0; k < neural.Count; k++)
                    {
                        Layer l = neural.Find(x => x.layer_index == k);
                        l_out[k + 1] = l.goThrough(l_out[k]);
                    }

                    for(uint k = 0; k < l_out[l_out.Length - 1].Length; k++)
                    {
                        iter_res[j] += Math.Pow(l_out[l_out.Length-1][k] - (double)output_data[j][k], 2);
                    }
                } 



            }
        }

        public void add_layer(int in_size, int out_size)
        {
            neural.Add(new Layer(in_size, out_size, (uint)neural.Count));
        }

        public class Layer
        {
            public uint layer_index;
            double[][] weight;
            double[] bias;

            public int out_size;

            public Layer(int inputs_size, int output_size, uint index = 0)
            {
                layer_index = index;
                weight = new double[output_size][];
                bias = new double[output_size];
                out_size = output_size;
                Random random = new Random();

                for (int i = 0; i < weight.Length; i++)
                {
                    weight[i] = new double[inputs_size];
                    for(int j = 0; j < weight[i].Length; j++) weight[i][j] = 1;
                }

                Array.Clear(bias, 0, bias.Length);
            }

            public  double[] goThrough(double[] input)
            {
                double[] out_array = new double[out_size];
                double buf = 0;

                for(uint i = 0; i < weight.Length; i++)
                {
                    for(uint j = 0; j < input.Length; j++)
                    {
                        buf += input[j] * weight[i][j];
                    }

                    out_array[i] = Math.Tanh(buf - bias[i]);
                }

                return out_array;
            }
        }
    }
}
