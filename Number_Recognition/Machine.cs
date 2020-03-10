using System;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System.Collections.Generic;

namespace Number_Recognition
{
    class Machine
    {
        private double[][] output_data;
        private double[][] input_data;
        private double learn_rate; //Coefficient to be applied to the back propagation
        private uint batch_size; //Size of the batch
        private uint data_length; //Number of elements to feed to the network
        private uint result_size;

        //List of the layers
        private List<Layer> neural_net_layers_list = new List<Layer>();

        private double[][] normalize(byte[][] data, double val)
        {
            double[][] data_out = new double[data.Length][];
            for(int i = 0; i < data.Length; i++)
            {
                data_out[i] = new double[data[i].Length];

                for(int j = 0; j < data[i].Length; j++)
                {
                    data_out[i][j] = (double)data[i][j] / val;
                }
            }

            return data_out;
        }

        public Machine(byte[] labels, Func<byte[], double[][]> init_result_array, byte[][] values, uint result_size, double learning_rate = 0.01, uint batch_size = 100, double normalizing_value = 255)
        {
            learn_rate = learning_rate;
            this.batch_size = batch_size;
            this.result_size = result_size;

            input_data = normalize(values, normalizing_value);
            data_length = (uint)values.Length;
            output_data = init_result_array(labels);
        }
        /// <summary>
        /// Use this to feed the training data to the network. The training will stop once the number of iterations is
        /// achieved.
        /// </summary>
        /// <param name="iterations"></param>
        public void train(uint iterations)
        {
            for(uint i = 0; i < iterations; i++ )
            {
                Vector<double> cost_values = DenseVector.Create((int)data_length, 0);

                //Feed all the training data to the network
                for (uint j = 0; j < data_length; j++)
                {
                    //Data 
                    Vector<double> data = DenseVector.Build.Dense(input_data[j]);
                    //Result to be obtained
                    Vector<double> result = DenseVector.Build.Dense(output_data[j]);
                    //
                    Vector<double> network_out;

                    network_out = foward_propagate(data);

                    cost_values[(int)j] = (result - network_out).Map(elem => Math.Pow(elem, 2)).Sum();

                    
                }

                Console.WriteLine(cost_values);
            }
        }

        public void add_layer(int in_size, int out_size, Layer.SQAUSH_FUNC func = Layer.SQAUSH_FUNC.TANH)
        {
            neural_net_layers_list.Add(new Layer(in_size, out_size, func, (uint)neural_net_layers_list.Count));
        }

        private static int layer_num = 0;
        private Vector<double> res;
        private Vector<double> foward_propagate(Vector<double> start_data)
        {
            if(layer_num < neural_net_layers_list.Count)
            {
                Layer l = neural_net_layers_list.Find(x => x.layer_index == layer_num);
                layer_num++;
                res = l.propagate(start_data);
                foward_propagate(res);
            }

            return res;
        }

        private void backPropagate(Vector<double> expeted)
        {
           for(int i = neural_net_layers_list.Count; i > 0; --i)
           {
                Layer l = neural_net_layers_list.Find(x => x.layer_index == i);

                //l.global_error(l.activations_L, expeted, l.activations_Lminus1, )
           }
        }

        public class Layer
        {
            public enum SQAUSH_FUNC {TANH, RELU, SIGMOID};
            SQAUSH_FUNC func;
            public uint layer_index;

            private Matrix<double> weights;
            public Vector<double> activations_L { get; private set; }
            public Vector<double> activations_Lminus1 { get; private set; }
            private Vector<double> bias;

            public int out_size;

            private double squash(double val)
            {
                switch(func)
                {
                    case SQAUSH_FUNC.TANH:
                        return Math.Tanh(val);
                    case SQAUSH_FUNC.SIGMOID:
                        return (1 / (1 + Math.Exp(-val)));
                    default:
                        return Math.Tanh(val);
                }
            }

            private double derivative_squash(double val)
            {
                switch(this.func)
                {
                    case SQAUSH_FUNC.TANH:
                        return (1 - Math.Tanh(val));
                    case SQAUSH_FUNC.SIGMOID:
                        return (squash(val) * (1 - squash(val)));
                    default:
                        return (1 - Math.Tanh(val));
                }
            }

            private double delta_w(double out_effective, double out_expected, double act_Lminus1, double w_L, double b_L)
            {
                return act_Lminus1 * global_error(out_effective, out_expected, act_Lminus1, w_L, b_L);
            }

            public double global_error(double out_effective, double out_expected, double act_Lminus1, double w_L, double b_L)
            {
                return 2 * (out_effective - out_expected) * derivative_squash(w_L * act_Lminus1 + b_L);
            }

            public Layer(int inputs_size, int output_size, SQAUSH_FUNC func, uint index = 0)
            {
                this.func = func;
                layer_index = index;
                Random random = new Random();

                activations_L = DenseVector.Create(output_size, 0);
                activations_Lminus1 = DenseVector.Create(inputs_size, 0);
                weights = DenseMatrix.Build.Dense(output_size, inputs_size, (i, j) => 0);
                bias = DenseVector.Build.Dense(output_size, (i) => 0);

                out_size = output_size;
            }

            public  Vector<double> propagate(Vector<double> input)
            {
                activations_Lminus1 = input;
                //Sum of the weight and bias
                activations_L = weights * activations_Lminus1 + bias;

                //squash all the elements of the resulating vector
                activations_L.Map((i) => squash((i)), activations_L, Zeros.AllowSkip);
                return activations_L;
            }

            public void backPropagate()
            {
               
            }
        }
    }
}
