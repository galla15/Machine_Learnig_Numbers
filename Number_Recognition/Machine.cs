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

        public Machine(byte[] labels, byte[][] values, uint result_size, Func<byte[], double[][]> init_result_array = null, double learning_rate = 0.01,
                        uint batch_size = 100, double normalizing_value = 255)
        {

            learn_rate = learning_rate;
            this.batch_size = batch_size;
            this.result_size = result_size;
            input_data = normalize(values, normalizing_value);
            data_length = (uint)values.Length;
            if (init_result_array != null) output_data = init_result_array(labels);
        }
        /// <summary>
        /// Use this to feed the training data to the network. The training will stop once the number of iterations is
        /// achieved.
        /// </summary>
        /// <param name="iterations"></param>
        public void train(uint iterations)
        {
            double err = 0;
            uint iter = 0;
            for(uint i = 0; i < iterations; i++ )
            {
                double[] cost_values = new double[data_length];

                //Feed all the training data to the network
                for (uint j = 0; j < data_length; j++)
                {
                    //Data 
                    Vector<double> data = DenseVector.Build.Dense(input_data[j]);
                    //Result to be obtained
                    Vector<double> result = DenseVector.Build.Dense(output_data[j]);
                    //Output from the network
                    Vector<double> network_out;

                    network_out = foward_propagate(data);

                    back_Propagate(result - network_out);

                    cost_values[j] = (result - network_out).Map(x => Math.Pow(x, 2) * 0.5).Sum() / result.Count;
                      
                }
                err = 0;

                foreach (double d in cost_values)
                {
                    err += d;
                }
                err = err / cost_values.Length;

                iter = i;

                if (err < 0.1)
                {
                    break;
                }

            }

            Console.WriteLine("Error : {0}, iterations : {1}", err, iter);
        }

        public void add_layer(int in_size, int out_size, Squash_func.SQUASH_FUNC func = Squash_func.SQUASH_FUNC.TANH)
        {
            neural_net_layers_list.Add(new Layer(in_size, out_size, func, (uint)neural_net_layers_list.Count, learn_rate: this.learn_rate));
        }

        
        private Vector<double> foward_propagate(Vector<double> start_data)
        {
            Vector<double> res = start_data;

            for(int i = 0; i < neural_net_layers_list.Count; i++)
            {
                Layer l = neural_net_layers_list.Find(x => x.layer_index == i);
                res = l.propagate(start_data);
            }
            
            return res;
        }

        private void back_Propagate(Vector<double> error)
        {
            Vector<double> err = error;

            for(int i = neural_net_layers_list.Count - 1; i > -1; i--)
            {
                Layer l = neural_net_layers_list.Find(x => x.layer_index == i);
                err = l.backPropagate(err);
            }

        }

        public static class Squash_func
        {
            public enum SQUASH_FUNC { TANH, SIGMOID, RELU};

            public static double squash(SQUASH_FUNC func, double val)
            {
                switch (func)
                {
                    case SQUASH_FUNC.TANH:
                        return Math.Tanh(val);
                    case SQUASH_FUNC.SIGMOID:
                        return (1 / (1 + Math.Exp(-val)));
                    default:
                        return Math.Tanh(val);
                }
            }

            public static double derivative_squash(SQUASH_FUNC func, double val)
            {
                switch (func)
                {
                    case SQUASH_FUNC.TANH:
                        return (1 - Math.Tanh(val));
                    case SQUASH_FUNC.SIGMOID:
                        return (squash(func, val) * (1 - squash(func, val)));
                    default:
                        return (1 - Math.Tanh(val));
                }
            }
        }

        private class Neuron
        {
            Func<double, double> squash_func;
            Func<double, double> squash_func_derivate;
            Vector<double> weigths = null;
            double bias;
            double output;
            double learning_rate;

            bool forward_propagation_done;

            public Neuron(Squash_func.SQUASH_FUNC func, int input_size, double learn_rate)
            {
                //Init the squash function
                squash_func = new Func<double, double>(x => Squash_func.squash(func, x));
                //Init the derivative of the squash function
                squash_func_derivate = new Func<double, double>(x => Squash_func.derivative_squash(func, x));
                Random random = new Random();

                //Initialize the weights of the neuron to random value between 0 and 1
                weigths = DenseVector.Build.Dense(input_size).Map(x => random.NextDouble());

                //Initilize the bias to a random value between 0 and 1
                bias = random.NextDouble();

                forward_propagation_done = false;
                learning_rate = learn_rate;
            }
            public double foward_propagate(Vector<double> inputs)
            {
                forward_propagation_done = true;
                output = squash_func(inputs * weigths + bias);
                return output;
            }

            public Vector<double> back_propagate_error(double error, Vector<double> inputs)
            {
                Vector<double> error_next_layer;

                if (forward_propagation_done)
                {
                    double delta_b = error * squash_func_derivate(output);
                    bias += learning_rate * delta_b;

                    weigths += learning_rate * delta_b * inputs;

                    error_next_layer = weigths.Map(x => x * delta_b);

                    forward_propagation_done = false;

                    return error_next_layer;

                }
                else throw new Exception("Forward propagation must be called before doing the back error propagation!");
            }
           
        }

        private class Layer
        {
            Neuron[] neurons;
            Vector<double> inputs;
            Vector<double> outputs;
            int out_size;
            int input_size;
            public uint layer_index { get; private set; }

            public Layer(int inputs_size, int output_size, Squash_func.SQUASH_FUNC func, uint index = 0, double learn_rate = 0.01)
            {
                out_size = output_size;
                input_size = inputs_size;
                
                neurons = new Neuron[out_size];
                for(int i = 0; i < neurons.Length; i++)
                {
                    neurons[i] = new Neuron(func, inputs_size, learn_rate);
                }

                inputs = DenseVector.Build.Dense(input_size, 0);
                outputs = DenseVector.Build.Dense(out_size, 0);

                layer_index = index;
            }

            public  Vector<double> propagate(Vector<double> input)
            {
                inputs = input;

                foreach(Neuron n in neurons)
                {
                    outputs[Array.FindIndex(neurons, x => x == n)] = n.foward_propagate(inputs);
                }

                return outputs;
            }

            public Vector<double> backPropagate(Vector<double> error)
            {
                Vector<double> error_next_layer = DenseVector.Build.Dense(input_size,0);
                int count = 0;

                foreach (Neuron n in neurons)
                {
                    count++;
                    double err = error[Array.FindIndex(neurons, x => x == n)];
                    error_next_layer += n.back_propagate_error(err, inputs);
                }

                error_next_layer.Map(x => x / count);

                return error_next_layer;

            }

        }
    }
}
