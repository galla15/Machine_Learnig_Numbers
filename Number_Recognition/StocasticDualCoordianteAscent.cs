using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Remoting.Contexts;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Number_Recognition
{
    class StocasticDualCoordianteAscent
    {
        private MLContext context = new MLContext();
        private IDataView data;

        private class Digit
        {
            [VectorType(785)] public float[] PixelValues;
        }

        private class DigitPrediction
        {
            public float[] Score;
        }

        public class _data
        {
            [ColumnName("Features")]
            [VectorType(784)]
            public float[] Features { get; set; }
         
            [ColumnName("Label")]
            public float Label { get; set; }

            [ColumnName("Score")]
            public float Score { get; set; }

            [ColumnName("PredictedLabel")]
            public float PredictedLabel { get; set; }

            public _data(int size)
            {
                Features = new float[size];
            }
        }

        public static float double_to_float(double d)
        {
            return (float)d;
        }

        private static IEnumerable<_data> GetSampleData(double[][] inputs, double[] labels)
        {
            _data[] vs = new _data[inputs.Length];

            for (int i = 0; i < inputs.Length; i++)
            {
                vs[i] = new _data(inputs[i].Length);
                Array.Copy(Array.ConvertAll(inputs[i], new Converter<double, float>(double_to_float)), vs[i].Features, inputs[i].Length);
                vs[i].Label = (float)labels[i];
            }

            return vs;
        }



        public StocasticDualCoordianteAscent(double[][] inputs, double[] labels)
        {

            IDataView data_in = context.Data.LoadFromEnumerable<_data>(GetSampleData(inputs, labels));

            DataOperationsCatalog.TrainTestData partitions = context.Data.TrainTestSplit(data_in);

            Microsoft.ML.Transforms.ColumnConcatenatingEstimator pipeline = context.Transforms.Concatenate("Features", nameof(_data.Features));

            pipeline.AppendCacheCheckpoint(context);

            pipeline.Append(context.MulticlassClassification.Trainers.SdcaNonCalibrated());

            ColumnConcatenatingTransformer model = pipeline.Fit(partitions.TrainSet);

            //var engine = ModelOperationsCatalog.CreatePredictionEngine<Digit, DigitPrediction>(model);
            Console.WriteLine("Evaluating model....");
            IDataView predictions = model.Transform(partitions.TestSet);

            // evaluate the predictions
            MulticlassClassificationMetrics metrics = context.MulticlassClassification.Evaluate(predictions);

            // show evaluation metrics
            Console.WriteLine($"Evaluation metrics");
            Console.WriteLine($"    MicroAccuracy:    {metrics.MicroAccuracy:0.###}");
            Console.WriteLine($"    MacroAccuracy:    {metrics.MacroAccuracy:0.###}");
            Console.WriteLine($"    LogLoss:          {metrics.LogLoss:#.###}");
            Console.WriteLine($"    LogLossReduction: {metrics.LogLossReduction:#.###}");
            Console.WriteLine();
        }

    }
}
