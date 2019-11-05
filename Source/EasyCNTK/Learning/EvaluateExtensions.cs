//
// Copyright (c) Stanislav Grigoriev. All rights reserved.
// grigorievstas9@gmail.com 
// https://github.com/StanislavGrigoriev/EasyCNTK
//
// Copyright (c) Microsoft. All rights reserved.
//
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//
using CNTK;
using EasyCNTK.Learning.Metrics;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

namespace EasyCNTK.Learning
{
    public static class EvaluateExtensions
    {
        /// <summary>
        /// Returns metrics for regression tasks. If the target variable is multi-dimensional, metrics are returned for each dimension independently.
        /// </summary>
        /// <typeparam name="T">Data type. Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <returns></returns>
        public static IList<RegressionMetrics> GetRegressionMetrics<T>(this IEnumerable<EvaluateItem<T>> source) where T: IConvertible
        {
            var firstItem = source.FirstOrDefault();
            if (firstItem.Equals(default(EvaluateItem<T>)))
            {
                throw new ArgumentException("Последовательность IEnumerable<EvaluateItem<T>> не содержит элементов.", "source");
            }
            
            var result = firstItem.EvaluatedValue
                .Select(p => new RegressionMetrics())
                .ToArray();
           
            var expectedDataAccumulator = new double[result.Length];
            int countItems = 0;
            foreach (var item in source)
            {
                for (int i = 0; i < result.Length; i++)
                {
                    double evaluated = item.EvaluatedValue[i].ToDouble(CultureInfo.InvariantCulture);
                    double expected  = item.ExpectedValue[i].ToDouble(CultureInfo.InvariantCulture);
                    double mae       = Math.Abs(evaluated - expected);
                    double rmse      = Math.Pow(evaluated - expected, 2);
                    checked
                    {
                        result[i].MAE += mae;
                        result[i].RMSE += rmse;                        
                        expectedDataAccumulator[i] += expected;
                    }
                }

                countItems++;
            }
            for (int i = 0; i < result.Length; i++)
            {
                expectedDataAccumulator[i] = expectedDataAccumulator[i] / countItems;
            }

            var expectedVarianceAccumulator = new double[result.Length];
            foreach (var item in source)
            {
                for (int i = 0; i < result.Length; i++)
                {
                    double expected = item.ExpectedValue[i].ToDouble(CultureInfo.InvariantCulture);
                    checked
                    {
                        expectedVarianceAccumulator[i] += Math.Pow(expected - expectedDataAccumulator[i], 2);
                    }
                }
            }

            for (int i = 0; i < result.Length; i++)
            {
                result[i].Determination = 1 - (result[i].RMSE / expectedVarianceAccumulator[i]);
                result[i].MAE = result[i].MAE / countItems;
                result[i].RMSE = Math.Sqrt(result[i].RMSE / countItems);
                
            }

            return result;
        }
        /// <summary>
        /// Calculates metrics for binary classification problems. 
        /// It is understood that the output has unit dimension and class labels are encoded at 1 for True observations, at 0 for False.
        /// </summary>
        /// <typeparam name="T">Data type. Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="threshold">The threshold value for the actual output value of the neural network, below which the class is defined as False.</param>
        /// /// <returns></returns>
        public static BinaryClassificationMetrics GetBinaryClassificationMetrics<T>(this IEnumerable<EvaluateItem<T>> source, double threshold = 0.5) where T: IConvertible
        {
            var firstItem = source.FirstOrDefault();
            if (firstItem.Equals(default(EvaluateItem<T>)))
            {
                throw new ArgumentException("Последовательность IEnumerable<EvaluateItem<T>> не содержит элементов.", "source");
            }

            int TP = 0; //fact 1, score 1
            int TN = 0; //fact 0, score 0
            int FP = 0; //fact 0, score 1
            int FN = 0; //fact 1, score 0

            foreach (var item in source)
            {                
                int expected = item.ExpectedValue[0].ToInt32(CultureInfo.InvariantCulture);
                int evaluated = item.EvaluatedValue[0].ToDouble(CultureInfo.InvariantCulture) < threshold ? 0 : 1;

                bool isPositive = expected == 1;
                if (isPositive)
                {
                    if (expected == evaluated)
                    {
                        TP++; 
                    }
                    else
                    {
                        FN++;
                    }
                }
                else
                {
                    if (expected == evaluated)
                    {
                        TN++;
                    }
                    else
                    {
                        FP++;
                    }
                }
            }

            double countSamples = TP + TN + FP + FN;
            var accuracy = (TP + TN) / countSamples;
            var precision = (double)TP / (TP + FP);
            var recall = (double)TP / (TP + FN);
            var f1score = 2 * precision * recall / (precision + recall);
            var confusionMatrix = new double[2, 2]
            {
                { TP/countSamples, FP/countSamples },
                { FN/countSamples, TN/countSamples }
            };

            return new BinaryClassificationMetrics()
            {
                Accuracy = accuracy,
                Precision = precision,
                Recall = recall,
                F1Score = f1score,
                ConfusionMatix = confusionMatrix
            };
        }
        /// <summary>
        /// Computes metrics for single-class classification problems. 
        /// It is understood that the output is encoded in One-Hot-Encoding (and wrapped in Softmax, although it is possible to use <seealso cref="ActivationFunctions.Sigmoid"/> , <seealso cref="ActivationFunctions.HardSigmoid"/> ), otherwise the metric is not calculated correctly.
        /// </summary>
        /// <typeparam name="T">Data type. Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <returns></returns>
        public static OneLabelClassificationMetrics GetOneLabelClassificationMetrics<T>(this IEnumerable<EvaluateItem<T>> source) where T:IConvertible
        {
            var firstElement = source.FirstOrDefault();
            if (firstElement.Equals(default(EvaluateItem<T>)))
            {
                throw new ArgumentException("Последовательность IEnumerable<EvaluateItem<T>> не содержит элементов.", "source");
            }
            
            var confusionMatrix = new double[firstElement.EvaluatedValue.Count, firstElement.EvaluatedValue.Count];
            var classesDistribution = Enumerable.Range(0, firstElement.EvaluatedValue.Count)
                .Select(p => new ClassItem()
                {
                    Index = p
                })
                .ToList();
            int countAccurateSamples = 0;
            int countSamples = 0;
            foreach (var item in source)
            {
                int expected = item.ExpectedValue.IndexOf(item.ExpectedValue.Max());
                int evaluated = item.EvaluatedValue.IndexOf(item.EvaluatedValue.Max());

                classesDistribution[expected].Fraction++;
                confusionMatrix[expected, evaluated]++;
                if (expected == evaluated)
                {
                    countAccurateSamples++;                    
                    classesDistribution[evaluated].Recall++;
                }
                classesDistribution[evaluated].Precision++;
                countSamples++;
            }
            for (int i = 0; i < firstElement.EvaluatedValue.Count; i++)
            {
                classesDistribution[i].Precision = classesDistribution[i].Precision == 0 ? 0 : classesDistribution[i].Recall / classesDistribution[i].Precision;
                classesDistribution[i].Recall /= classesDistribution[i].Fraction;
                classesDistribution[i].F1Score = (classesDistribution[i].Precision + classesDistribution[i].Recall) == 0 ? 0
                    : 2 * classesDistribution[i].Precision * classesDistribution[i].Recall / (classesDistribution[i].Precision + classesDistribution[i].Recall);
                classesDistribution[i].Fraction /= countSamples;
                for (int j = 0; j < firstElement.EvaluatedValue.Count; j++) 
                {
                    confusionMatrix[i, j] /= countSamples;
                }
            }
                
            double accuracy = (double)countAccurateSamples / countSamples;
            return new OneLabelClassificationMetrics()
            {
                Accuracy = accuracy,
                ConfusionMatrix = confusionMatrix,
                ClassesDistribution = classesDistribution
            };
        }
        /// <summary>
        /// Computes metrics for multiclass classification problems.
        /// It is understood that the output is encoded in One-Hot-Encoding (and wrapped in <seealso cref="ActivationFunctions.Sigmoid"/> , <seealso cref="ActivationFunctions.HardSigmoid"/> etc.), otherwise the metric is not calculated correctly.
        /// </summary>
        /// <typeparam name="T">Data type. Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="thershold">The threshold value for the actual value of the output of the neural network, below which the class is not recognized. In other words, this is the minimum probability that the classifier must give for a particular class so that this class is considered as recognized.</param>
        /// <returns></returns>
        public static MultiLabelClassificationMetrics GetMultiLabelClassificationMetrics<T>(this IEnumerable<EvaluateItem<T>> source, double thershold = 0.5) where T:IConvertible
        {
            var firstElement = source.FirstOrDefault();
            if (firstElement.Equals(default(EvaluateItem<T>)))
            {
                throw new ArgumentException("Последовательность IEnumerable<EvaluateItem<T>> не содержит элементов.", "source");
            }
            
            var classesDistribution = Enumerable.Range(0, firstElement.EvaluatedValue.Count)
                .Select(p => new ClassItem()
                {
                    Index = p
                })
                .ToList();
            int countAccurateLabels = 0;
            int countLabels = 0;
            foreach (var item in source)
            {
                var expected = item.ExpectedValue
                    .Select((value, index) => value.ToDouble(CultureInfo.InvariantCulture) > thershold ? index : -1)
                    .Where(p => p != -1)
                    .ToList();
                var evaluated = item.EvaluatedValue
                    .Select((value, index) => value.ToDouble(CultureInfo.InvariantCulture) > thershold ? index : -1)
                    .Where(p => p != -1)
                    .ToList();

                foreach (var target in expected)
                {
                    classesDistribution[target].Fraction++;
                    if (evaluated.Contains(target))
                    {
                        classesDistribution[target].Recall++;
                    }
                    countLabels++;
                }
                evaluated.ForEach(evaluate => classesDistribution[evaluate].Precision++);
            }
            classesDistribution.ForEach(p =>
            {
                p.Precision = p.Precision == 0 ? 0 : p.Recall / p.Precision;
                p.Recall /= p.Fraction;
                p.F1Score = (p.Precision + p.Recall) == 0 ? 0 : 2 * p.Precision * p.Recall / (p.Precision + p.Recall);
                p.Fraction /= countLabels;
            });

            double accuracy = (double)countAccurateLabels / countLabels;
            return new MultiLabelClassificationMetrics()
            {
                Accuracy = accuracy,                
                ClassesDistribution = classesDistribution
            };
        }


        #region Function extensions

        #region Evaluate<T>
        /// <summary>
        /// Computes the model output for each of the test cases.
        /// </summary>
        /// <typeparam name="T">Data type. Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="testData">Test data. Each minipack must contain 1 test case.</param>
        /// <param name="device">Device for calculations</param>
        /// <param name="inputName">The name of the input layer. The name must be unique throughout the network. There can be several inputs, this parameter indicates which of them serves data.</param>
        /// <returns></returns>
        private static IEnumerable<EvaluateItem<T>> Evaluate<T>(this Function source,
            IEnumerable<Minibatch> testData,
            DeviceDescriptor device,
            string inputName) where T : IConvertible
        {
            Variable inputVariable;
            try
            {
                inputVariable = source.Inputs.Single(p => p.Name.ToUpper() == inputName.ToUpper());
            }
            catch (InvalidOperationException ex)
            {
                throw new InvalidOperationException("Модель имеет несколько входов с одинаковым именем.", ex);
            }               
            foreach (var miniBatch in testData)
            {
                var inputDataMap = new Dictionary<Variable, Value>() { { inputVariable, miniBatch.Features } };
                var outputDataMap = new Dictionary<Variable, Value>() { { source.Output, null } };

                source.Evaluate(inputDataMap, outputDataMap, device);

                var expected = miniBatch.Labels.GetDenseData<T>(source.Output);
                var evaluated = outputDataMap[source.Output].GetDenseData<T>(source.Output);

                foreach (var item in expected.Zip(evaluated, (exp, eval) => (exp, eval)))
                {
                    var evaluateItem = new EvaluateItem<T>(item.exp, item.eval);
                    yield return evaluateItem;
                }
            }
        }
        private static IEnumerable<EvaluateItem<T>[]> Evaluate<T>(this Function source,
            IEnumerable<MinibatchMultiOutput> testData,
            DeviceDescriptor device,
            string inputName) where T : IConvertible
        {
            Variable inputVariable;
            try
            {
                inputVariable = source.Inputs.Single(p => p.Name.ToUpper() == inputName.ToUpper());
            }
            catch (InvalidOperationException ex)
            {
                throw new InvalidOperationException("Модель имеет несколько входов с одинаковым именем.", ex);
            }           
            foreach (var miniBatch in testData)
            {
                var inputDataMap = new Dictionary<Variable, Value>() { { inputVariable, miniBatch.Features } };

                var result = new (IList<IList<T>> expected, IList<IList<T>> evaluated)[miniBatch.Labels.Length];               
                for (int i = 0; i < result.Length; i++)
                {                    
                    var outputDataMap = new Dictionary<Variable, Value>() { { source.Outputs[i], null } };

                    source.Evaluate(inputDataMap, outputDataMap, device);

                    var expected = miniBatch.Labels[i].GetDenseData<T>(source.Output);
                    var evaluated = outputDataMap[source.Outputs[i]].GetDenseData<T>(source.Output);

                    result[i] = (expected, evaluated);
                }

                for (int i = 0; i < miniBatch.Size; i++)
                {
                    yield return result
                        .Select(p => new EvaluateItem<T>(p.expected[i], p.evaluated[i]))
                        .ToArray();                        
                }
            }
        }

        #endregion

        #region Predict<T>

        #region One input - One output
        /// <summary>
        /// Calculates the output model values for each of the input examples (user example). (Inference).
        /// </summary>
        /// <typeparam name="T">Data type. Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="data">A set of examples for which the output is calculated</param>
        /// <param name="device">Device for calculations</param>
        /// <param name="inputName">The name of the input layer. The name must be unique throughout the network. There can be several inputs, this parameter indicates which of them serves data.</param>
        /// <returns></returns>
        public static IEnumerable<T[]> Predict<T>(this Function source,
           IEnumerable<Value> data,
           DeviceDescriptor device,
           string inputName) where T : IConvertible
        {
            Variable inputVariable;
            try
            {
                inputVariable = source.Inputs.Single(p => p.Name.ToUpper() == inputName.ToUpper());
            }
            catch (InvalidOperationException ex)
            {
                throw new InvalidOperationException("Модель имеет несколько входов с одинаковым именем.", ex);
            }           
            foreach (var features in data)
            {
                var inputDataMap = new Dictionary<Variable, Value>() { { inputVariable, features } };
                var outputDataMap = new Dictionary<Variable, Value>() { { source.Output, null } };

                source.Evaluate(inputDataMap, outputDataMap, device);

                var predicted = outputDataMap[source.Output].GetDenseData<T>(source.Output);

                foreach (var item in predicted)
                {
                    yield return item.ToArray();
                }
            }
        }
        /// <summary>
        /// Calculates the output model values for each of the input examples. (Inference).
        /// </summary>
        /// <typeparam name="T">Data type. Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="data">A set of examples for which the output is calculated</param>
        /// <param name="device">Device for calculations</param>
        /// <param name="inputName">The name of the input layer. The name must be unique throughout the network. There can be several inputs, this parameter indicates which of them serves data.</param>
        /// <returns></returns>
        public static IEnumerable<T[]> Predict<T>(this Function source,
            IEnumerable<T[]> data,
            DeviceDescriptor device,
            int minibatchSize = 512,
            string inputName = "Input") where T : IConvertible
        {
            DataConverter dataConverter = new DataConverter(device);
            var values = dataConverter.ConvertDataToValue(data, minibatchSize);
            return source.Predict<T>(values, device, inputName);
        }
        /// <summary>
        /// Calculates the output model values for each of the input examples (example is a sequence). (Inference).
        /// </summary>
        /// <typeparam name="T">Data type. Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="device">Device for calculations</param>
        /// <param name="inputName">The name of the input layer. The name must be unique throughout the network. There can be several inputs, this parameter indicates which of them serves data.</param>
        /// <returns></returns>
        public static IEnumerable<T[]> Predict<T>(this Function source,
            IEnumerable<IList<T[]>> data,
            DeviceDescriptor device,
            int minibatchSize = 512,
            string inputName = "Input") where T : IConvertible
        {
            DataConverter dataConverter = new DataConverter(device);
            var values = dataConverter.ConvertDataToValue(data, minibatchSize);
            return source.Predict<T>(values, device, inputName);
        }
        /// <summary>
        /// Calculates the output model values for each of the input examples (example - 2D). (Inference).
        /// </summary>
        /// <typeparam name="T">Data type. Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="device">Device for calculations</param>
        /// <param name="inputName">The name of the input layer. The name must be unique throughout the network. There can be several inputs, this parameter indicates which of them serves data.</param>
        /// <returns></returns>
        public static IEnumerable<T[]> Predict<T>(this Function source,
            IEnumerable<T[,]> data,
            DeviceDescriptor device,
            int minibatchSize = 512,
            string inputName = "Input") where T : IConvertible
        {
            DataConverter dataConverter = new DataConverter(device);
            var values = dataConverter.ConvertDataToValue(data, minibatchSize);
            return source.Predict<T>(values, device, inputName);
        }

        /// <summary>
        /// Computes the output values of the model of one example (user-defined example). (Inference).
        /// </summary>
        /// <typeparam name="T">Data type. Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="device">Device for calculations</param>
        /// <param name="inputName">The name of the input layer. The name must be unique throughout the network. There can be several inputs, this parameter indicates which of them serves data.</param>
        /// <returns></returns>
        public static T[] Predict<T>(this Function source, Value data, DeviceDescriptor device, string inputName = "Input") where T : IConvertible
        {
            return source.Predict<T>(Enumerable.Repeat(data, 1), device, inputName).FirstOrDefault();
        }
        /// <summary>
        /// Computes the output values of the model of one example (example - sequence). (Inference).
        /// </summary>
        /// <typeparam name="T">Data type. Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="device">Device for calculations</param>
        /// <param name="inputName">The name of the input layer. The name must be unique throughout the network. There can be several inputs, this parameter indicates which of them serves data.</param>
        /// <returns></returns>
        public static T[] Predict<T>(this Function source, T[] data, DeviceDescriptor device, string inputName = "Input") where T : IConvertible
        {
            return source.Predict<T>(Enumerable.Repeat(data, 1), device, 1, inputName).FirstOrDefault();
        }
        /// <summary>
        /// Computes the output model values of one example. (Inference).
        /// </summary>
        /// <typeparam name="T">Data type. Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="device">Device for calculations</param>
        /// <param name="inputName">The name of the input layer. The name must be unique throughout the network. There can be several inputs, this parameter indicates which of them serves data.</param>
        /// <returns></returns>
        public static T[] Predict<T>(this Function source, IList<T[]> data, DeviceDescriptor device, string inputName = "Input") where T : IConvertible
        {
            return source.Predict<T>(Enumerable.Repeat(data, 1), device, 1, inputName).FirstOrDefault();
        }
        /// <summary>
        /// Computes the output values of the model of one example (example - 2D). (Inference).
        /// </summary>
        /// <typeparam name="T">Data type. Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="device">Device for calculations</param>
        /// <param name="inputName">The name of the input layer. The name must be unique throughout the network. There can be several inputs, this parameter indicates which of them serves data.</param>
        /// <returns></returns>
        public static T[] Predict<T>(this Function source, T[,] data, DeviceDescriptor device, string inputName = "Input") where T : IConvertible
        {
            return source.Predict<T>(Enumerable.Repeat(data, 1), device, 1, inputName).FirstOrDefault();
        }
        #endregion

        #region One input - Multi output
        /// <summary>
        /// Computes the output values of a model with multiple outputs for each of the input examples (custom example). (Inference).
        /// </summary>
        /// <typeparam name="T">Data type. Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="data">A set of examples for which the output is calculated</param>
        /// <param name="device">Device for calculations</param>
        /// <param name="inputName">The name of the input layer. The name must be unique throughout the network. There can be several inputs, this parameter indicates which of them serves data.</param>
        /// <returns></returns>
        public static IEnumerable<T[][]> PredictMultiOutput<T>(this Function source,
           IEnumerable<Value> data,
           DeviceDescriptor device,
           string inputName) where T : IConvertible
        {
            Variable inputVariable;
            try
            {
                inputVariable = source.Inputs.Single(p => p.Name.ToUpper() == inputName.ToUpper());
            }
            catch (InvalidOperationException ex)
            {
                throw new InvalidOperationException("Модель имеет несколько входов с одинаковым именем.", ex);
            }
            foreach (var features in data)
            {
                var inputDataMap = new Dictionary<Variable, Value>() { { inputVariable, features } };

                var result = new IList<IList<T>>[source.Outputs.Count];
                for (int i = 0; i < result.Length; i++)
                {
                    var outputDataMap = new Dictionary<Variable, Value>() { { source.Outputs[i], null } };

                    source.Evaluate(inputDataMap, outputDataMap, device);

                    result[i] = outputDataMap[source.Outputs[i]].GetDenseData<T>(source.Output);
                }
                //the network can immediately evaluate the entire minibatch of examples, so we go over the minibatch to give the result for each example
                for (int i = 0; i < result[0].Count; i++)
                {
                    yield return result
                        .Select(p => p[i].ToArray())
                        .ToArray();
                }
            }
        }

        /// <summary>
        /// Calculates the output model values for each of the input examples. (Inference).
        /// </summary>
        /// <typeparam name="T">Data type. Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="data">A set of examples for which the output is calculated</param>
        /// <param name="device">Device for calculations</param>
        /// <param name="inputName">The name of the input layer. The name must be unique throughout the network. There can be several inputs, this parameter indicates which of them serves data.</param>
        /// <returns></returns>
        public static IEnumerable<T[][]> PredictMultiOutput<T>(this Function source,
            IEnumerable<T[]> data,
            DeviceDescriptor device,
            int minibatchSize = 512, 
            string inputName = "Input") where T : IConvertible
        {
            DataConverter dataConverter = new DataConverter(device);
            var values = dataConverter.ConvertDataToValue(data, minibatchSize);
            return source.PredictMultiOutput<T>(values, device, inputName);
        }
        /// <summary>
        /// Calculates the output model values for each of the input examples (example is a sequence). (Inference).
        /// </summary>
        /// <typeparam name="T">Data type. Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="device">Device for calculations</param>
        /// <param name="inputName">The name of the input layer. The name must be unique throughout the network. There can be several inputs, this parameter indicates which of them serves data.</param>
        /// <returns></returns>
        public static IEnumerable<T[][]> PredictMultiOutput<T>(this Function source,
            IEnumerable<IList<T[]>> data,
            DeviceDescriptor device,
            int minibatchSize = 512,
            string inputName = "Input") where T : IConvertible
        {
            DataConverter dataConverter = new DataConverter(device);
            var values = dataConverter.ConvertDataToValue(data, minibatchSize);
            return source.PredictMultiOutput<T>(values, device, inputName);
        }
        /// <summary>
        /// Calculates the output model values for each of the input examples (example - 2D). (Inference).
        /// </summary>
        /// <typeparam name="T">Data type. Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="device">Device for calculations</param>
        /// <param name="inputName">The name of the input layer. The name must be unique throughout the network. There can be several inputs, this parameter indicates which of them serves data.</param>
        /// <returns></returns>
        public static IEnumerable<T[][]> PredictMultiOutput<T>(this Function source,
            IEnumerable<T[,]> data,
            DeviceDescriptor device,
            int minibatchSize = 512,
            string inputName = "Input") where T : IConvertible
        {
            DataConverter dataConverter = new DataConverter(device);
            var values = dataConverter.ConvertDataToValue(data, minibatchSize);
            return source.PredictMultiOutput<T>(values, device, inputName);
        }

        /// <summary>
        /// Computes the output values of the model of one example (user-defined example). (Inference).
        /// </summary>
        /// <typeparam name="T">Data type. Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="device">Device for calculations</param>
        /// <param name="inputName">The name of the input layer. The name must be unique throughout the network. There can be several inputs, this parameter indicates which of them serves data.</param>
        /// <returns></returns>
        public static T[][] PredictMultiOutput<T>(this Function source, Value data, DeviceDescriptor device, string inputName = "Input") where T : IConvertible
        {
            return source.PredictMultiOutput<T>(Enumerable.Repeat(data, 1), device, inputName).FirstOrDefault();
        }
        /// <summary>
        /// Computes the output values of the model of one example (example - sequence). (Inference).
        /// </summary>
        /// <typeparam name="T">Data type. Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="device">Device for calculations</param>
        /// <param name="inputName">The name of the input layer. The name must be unique throughout the network. There can be several inputs, this parameter indicates which of them serves data.</param>
        /// <returns></returns>
        public static T[][] PredictMultiOutput<T>(this Function source, T[] data, DeviceDescriptor device, string inputName = "Input") where T : IConvertible
        {
            return source.PredictMultiOutput<T>(Enumerable.Repeat(data, 1), device, 1, inputName).FirstOrDefault();
        }
        /// <summary>
        /// Computes the output model values of one example. (Inference).
        /// </summary>
        /// <typeparam name="T">Data type. Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="device">Device for calculations</param>
        /// <param name="inputName">The name of the input layer. The name must be unique throughout the network. There can be several inputs, this parameter indicates which of them serves data.</param>
        /// <returns></returns>
        public static T[][] PredictMultiOutput<T>(this Function source, IList<T[]> data, DeviceDescriptor device, string inputName = "Input") where T : IConvertible
        {
            return source.PredictMultiOutput<T>(Enumerable.Repeat(data, 1), device, 1, inputName).FirstOrDefault();
        }
        /// <summary>
        /// Computes the output values of the model of one example (example - 2D). (Inference).
        /// </summary>
        /// <typeparam name="T">Data type. Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="device">Device for calculations</param>
        /// <param name="inputName">The name of the input layer. The name must be unique throughout the network. There can be several inputs, this parameter indicates which of them serves data.</param>
        /// <returns></returns>
        public static T[][] PredictMultiOutput<T>(this Function source, T[,] data, DeviceDescriptor device, string inputName = "Input") where T : IConvertible
        {
            return source.PredictMultiOutput<T>(Enumerable.Repeat(data, 1), device, 1, inputName).FirstOrDefault();
        }
        #endregion

        #endregion

        #endregion

        #region Sequential extensions   

        #region Evaluate<T>
        /// <summary>
        /// Computes the model output for each of the test cases.
        /// </summary>
        /// <typeparam name="T">Data type. Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="testData">Test data. Each minipack must contain 1 test case.</param>
        /// <param name="device">Device for calculations</param>
        /// <param name="minibatchSize">The size of the mini-package for evaluation. Use allows you to evaluate data in batches (in parallel), without wasting resources on sending data to memory. The optimal size depends on the amount of data available on the GPU (better acceleration).</param>
        /// <param name="inputName">The name of the input layer. The name must be unique throughout the network. There can be several inputs, this parameter indicates which of them serves data.</param>
        /// <returns></returns>
        public static IEnumerable<EvaluateItem<T>> Evaluate<T>(this Sequential<T> source,
            IEnumerable<Minibatch> testData,
            DeviceDescriptor device,
            string inputName = "Input") where T : IConvertible
        {
            return source.Model.Evaluate<T>(testData, device, inputName);
        }
        /// <summary>
        /// Computes the model output for each of the test cases.
        /// </summary>
        /// <typeparam name="T">Data type. Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="testData">Test data set for training. Each example should contain signs at the beginning of the array with dimensions of inputDim, and at the end of class labels with dimensions of outputDim.
        /// For example, inputDim = 3, outputDim = 2: [f1, f2, f3, l1, l2]</param>
        /// <param name="inputDim">Dimension of signs (capacity)</param>
        /// <param name="device">Device for calculations</param>
        /// <param name="minibatchSize">The size of the mini-package for evaluation. Use allows you to evaluate data in batches (in parallel), without wasting resources on sending data to memory. The optimal size depends on the amount of data available on the GPU (better acceleration).</param>
        /// <param name="inputName">The name of the input layer. The name must be unique throughout the network. There can be several inputs, this parameter indicates which of them serves data.</param>
        /// <returns></returns>
        public static IEnumerable<EvaluateItem<T>> Evaluate<T>(this Sequential<T> source,
            IEnumerable<T[]> testData,
            int inputDim,
            DeviceDescriptor device,
            int minibatchSize = 512,
            string inputName = "Input") where T : IConvertible
        {
            DataConverter dataConverter = new DataConverter(device);
            var test = dataConverter.ConvertDatasetToMinibatch(testData, inputDim, minibatchSize);
            return source.Model.Evaluate<T>(test, device, inputName);
        }
        /// <summary>
        /// Computes the model output for each of the test cases.
        /// </summary>
        /// <typeparam name="T">Data type. Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="features">A set of test sequences (traits). Each sequence can be of variable length, but of the same dimension (the arrays of which the sequence consists must have the same length)</param>
        /// <param name="labels">A set of test marks. The dimension of the labels should be the same.</param>
        /// <param name="device">Device for calculations</param>
        /// <param name="minibatchSize">The size of the mini-package for evaluation. Use allows you to evaluate data in batches (in parallel), without wasting resources on sending data to memory. The optimal size depends on the amount of data available on the GPU (better acceleration).</param>
        /// <param name="inputName">The name of the input layer. The name must be unique throughout the network. There can be several inputs, this parameter indicates which of them serves data.</param>
        /// <returns></returns>
        public static IEnumerable<EvaluateItem<T>> Evaluate<T>(this Sequential<T> source,
            IEnumerable<IList<T[]>> features,
            IEnumerable<T[]> labels,
            DeviceDescriptor device,
            int minibatchSize = 512,
            string inputName = "Input") where T : IConvertible
        {
            DataConverter dataConverter = new DataConverter(device);
            var test = dataConverter.ConvertDatasetToMinibatch(features, labels, minibatchSize);
            return source.Model.Evaluate<T>(test, device, inputName);
        }
        /// <summary>
        /// Computes the model output for each of the test cases.
        /// </summary>
        /// <typeparam name="T">Data type. Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="features">Test data set</param>
        /// <param name="device">Device for calculations</param>
        /// <param name="minibatchSize">The size of the mini-package for evaluation. Use allows you to evaluate data in batches (in parallel), without wasting resources on sending data to memory. The optimal size depends on the amount of data available on the GPU (better acceleration).</param>
        /// <param name="inputName">The name of the input layer. The name must be unique throughout the network. There can be several inputs, this parameter indicates which of them serves data.</param>
        /// <returns></returns>
        public static IEnumerable<EvaluateItem<T>> Evaluate<T>(this Sequential<T> source,
            IEnumerable<T[,]> features,
            IEnumerable<T[]> labels,
            DeviceDescriptor device,
            int minibatchSize = 512,
            string inputName = "Input") where T : IConvertible
        {
            DataConverter dataConverter = new DataConverter(device);
            var test = dataConverter.ConvertDatasetToMinibatch(features, labels, minibatchSize);
            return source.Model.Evaluate<T>(test, device, inputName);
        }

        /// <summary>
        /// Computes the model output for each of the test cases.
        /// </summary>
        /// <typeparam name="T">Data type. Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="testData">Test data</param>
        /// <param name="device">Device for calculations</param>       
        /// <param name="inputName">The name of the input layer. The name must be unique throughout the network. There can be several inputs, this parameter indicates which of them serves data.</param>
        /// <returns></returns>
        public static IEnumerable<EvaluateItem<T>[]> Evaluate<T>(this SequentialMultiOutput<T> source,
            IEnumerable<MinibatchMultiOutput> testData,
            DeviceDescriptor device,
            string inputName = "Input") where T : IConvertible
        {
            return source.Model.Evaluate<T>(testData, device, inputName);
        }
        /// <summary>
        /// Computes the model output for each of the test cases.
        /// </summary>
        /// <typeparam name="T">Data type. Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="features">A set of signs.</param>
        /// <param name="labels">Set of labels. For each model output.</param>
        /// <param name="device">Device for calculations</param>
        /// <param name="minibatchSize">The size of the mini-package for evaluation. Use allows you to evaluate data in batches (in parallel), without wasting resources on sending data to memory. The optimal size depends on the amount of data available on the GPU (better acceleration).</param>
        /// <param name="inputName">The name of the input layer. The name must be unique throughout the network. There can be several inputs, this parameter indicates which of them serves data.</param>
        /// <returns></returns>
        public static IEnumerable<EvaluateItem<T>[]> Evaluate<T>(this SequentialMultiOutput<T> source,
            IEnumerable<T[]> features,
            IEnumerable<T[][]> labels,         
            DeviceDescriptor device,
            int minibatchSize = 512,
            string inputName = "Input") where T : IConvertible
        {
            DataConverter dataConverter = new DataConverter(device);
            var test = dataConverter.ConvertDatasetToMinibatchMultiOutput(features, labels,  minibatchSize);
            return source.Model.Evaluate<T>(test, device, inputName);
        }
        /// <summary>
        /// Computes the model output for each of the test cases.
        /// </summary>
        /// <typeparam name="T">Data type. Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="features">A set of test sequences (traits). Each sequence can be of variable length, but of the same dimension (the arrays of which the sequence consists must have the same length)</param>
        /// <param name="labels">Set of labels. For each model output.</param>
        /// <param name="device">Device for calculations</param>
        /// <param name="minibatchSize">The size of the mini-package for evaluation. Use allows you to evaluate data in batches (in parallel), without wasting resources on sending data to memory. The optimal size depends on the amount of data available on the GPU (better acceleration).</param>
        /// <param name="inputName">The name of the input layer. The name must be unique throughout the network. There can be several inputs, this parameter indicates which of them serves data.</param>
        /// <returns></returns>
        public static IEnumerable<EvaluateItem<T>[]> Evaluate<T>(this SequentialMultiOutput<T> source,
            IEnumerable<IList<T[]>> features,
            IEnumerable<T[][]> labels,
            DeviceDescriptor device,
            int minibatchSize = 512,
            string inputName = "Input") where T : IConvertible
        {
            DataConverter dataConverter = new DataConverter(device);
            var test = dataConverter.ConvertDatasetToMinibatchMultiOutput(features, labels, minibatchSize);
            return source.Model.Evaluate<T>(test, device, inputName);
        }
        /// <summary>
        /// Computes the model output for each of the test cases.
        /// </summary>
        /// <typeparam name="T">Data type. Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="features">Test data set</param>
        /// <param name="labels">Set of labels. For each model output.</param>
        /// <param name="device">Device for calculations</param>
        /// <param name="minibatchSize">The size of the mini-package for evaluation. Use allows you to evaluate data in batches (in parallel), without wasting resources on sending data to memory. The optimal size depends on the amount of data available on the GPU (better acceleration).</param>
        /// <param name="inputName">The name of the input layer. The name must be unique throughout the network. There can be several inputs, this parameter indicates which of them serves data.</param>
        /// <returns></returns>
        public static IEnumerable<EvaluateItem<T>[]> Evaluate<T>(this SequentialMultiOutput<T> source,
            IEnumerable<T[,]> features,
            IEnumerable<T[][]> labels,
            DeviceDescriptor device,
            int minibatchSize = 512,
            string inputName = "Input") where T : IConvertible
        {
            DataConverter dataConverter = new DataConverter(device);
            var test = dataConverter.ConvertDatasetToMinibatchMultiOutput(features, labels, minibatchSize);
            return source.Model.Evaluate<T>(test, device, inputName);
        }

        #endregion

        #region Predict<T>

        #region one input - one output
        /// <summary>
        /// Computes the output values of the model of one example (user-defined example). (Inference).
        /// </summary>
        /// <typeparam name="T">Data type. Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="device">Device for calculations</param>
        /// <param name="inputName">The name of the input layer. The name must be unique throughout the network. There can be several inputs, this parameter indicates which of them serves data.</param>
        /// <returns></returns>
        public static T[] Predict<T>(this Sequential<T> source, Value data, DeviceDescriptor device, string inputName = "Input") where T : IConvertible
        {
            return source.Model.Predict<T>(data, device, inputName);
        }
        /// <summary>
        /// Calculates the output model values for each of the input examples. (Inference).
        /// </summary>
        /// <typeparam name="T">Data type. Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="data">A set of examples for which the output is calculated</param>
        /// <param name="device">Device for calculations</param>
        /// <param name="inputName">The name of the input layer. The name must be unique throughout the network. There can be several inputs, this parameter indicates which of them serves data.</param>
        /// <returns></returns>
        public static IEnumerable<T[]> Predict<T>(this Sequential<T> source,
           IEnumerable<Value> data,
           DeviceDescriptor device,
           string inputName = "Input") where T : IConvertible
        {
            return source.Model.Predict<T>(data, device, inputName);
        }
        /// <summary>
        /// Calculates the output model values for each of the input examples. (Inference).
        /// </summary>
        /// <typeparam name="T">Data type. Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="data">A set of examples for which the output is calculated</param>
        /// <param name="device">Device for calculations</param>
        /// <param name="inputName">The name of the input layer. The name must be unique throughout the network. There can be several inputs, this parameter indicates which of them serves data.</param>
        public static IEnumerable<T[]> Predict<T>(this Sequential<T> source,
            IEnumerable<T[]> data,
            DeviceDescriptor device,
            int minibatchSize = 512,
            string inputName = "Input") where T : IConvertible
        {
            return source.Model.Predict<T>(data, device, minibatchSize, inputName);
        }
        /// <summary>
        /// Calculates the output model values for each of the input examples (example is a sequence). (Inference).
        /// </summary>
        /// <typeparam name="T">Data type. Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="device">Device for calculations</param>
        /// <param name="inputName">The name of the input layer. The name must be unique throughout the network. There can be several inputs, this parameter indicates which of them serves data.</param>
        /// <returns></returns>
        public static IEnumerable<T[]> Predict<T>(this Sequential<T> source,
            IEnumerable<IList<T[]>> data,
            DeviceDescriptor device,
            int minibatchSize = 512,
            string inputName = "Input") where T : IConvertible
        {
            return source.Model.Predict<T>(data, device, minibatchSize, inputName);
        }
        /// <summary>
        /// Calculates the output model values for each of the input examples (example - 2D). (Inference).
        /// </summary>
        /// <typeparam name="T">Data type. Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="device">Device for calculations</param>
        /// <param name="inputName">The name of the input layer. The name must be unique throughout the network. There can be several inputs, this parameter indicates which of them serves data.</param>
        /// <returns></returns>
        public static IEnumerable<T[]> Predict<T>(this Sequential<T> source,
            IEnumerable<T[,]> data,
            DeviceDescriptor device,
            int minibatchSize = 512,
            string inputName = "Input") where T : IConvertible
        {
            return source.Model.Predict<T>(data, device, minibatchSize, inputName);
        }
        /// <summary>
        /// Computes the output model values of one example. (Inference).
        /// </summary>
        /// <typeparam name="T">Data type. Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="device">Device for calculations</param>
        /// <param name="inputName">The name of the input layer. The name must be unique throughout the network. There can be several inputs, this parameter indicates which of them serves data.</param>
        /// <returns></returns>
        public static T[] Predict<T>(this Sequential<T> source, T[] data, DeviceDescriptor device, string inputName = "Input") where T : IConvertible
        {
            return source.Model.Predict<T>(data, device, inputName);
        }
        /// <summary>
        /// Computes the output values of the model of one example (example - sequence). (Inference).
        /// </summary>
        /// <typeparam name="T">Data type. Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="device">Device for calculations</param>
        /// <param name="inputName">The name of the input layer. The name must be unique throughout the network. There can be several inputs, this parameter indicates which of them serves data.</param>
        /// <returns></returns>
        public static T[] Predict<T>(this Sequential<T> source, IList<T[]> data, DeviceDescriptor device, string inputName = "Input") where T : IConvertible
        {
            return source.Model.Predict<T>(data, device, inputName);
        }
        /// <summary>
        /// Computes the output values of the model of one example (example - 2D). (Inference).
        /// </summary>
        /// <typeparam name="T">Data type. Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="device">Device for calculations</param>
        /// <param name="inputName">The name of the input layer. The name must be unique throughout the network. There can be several inputs, this parameter indicates which of them serves data.</param>
        /// <returns></returns>
        public static T[] Predict<T>(this Sequential<T> source, T[,] data, DeviceDescriptor device, string inputName = "Input") where T : IConvertible
        {
            return source.Model.Predict<T>(data, device, inputName);
        }
        #endregion

        #region one input - multi output
        /// <summary>
        /// Computes the output values of the model of one example (user-defined example). (Inference).
        /// </summary>
        /// <typeparam name="T">Data type. Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="device">Device for calculations</param>
        /// <param name="inputName">The name of the input layer. The name must be unique throughout the network. There can be several inputs, this parameter indicates which of them serves data.</param>
        /// <returns></returns>
        public static T[][] Predict<T>(this SequentialMultiOutput<T> source, Value data, DeviceDescriptor device, string inputName = "Input") where T : IConvertible
        {
            return source.Model.PredictMultiOutput<T>(data, device, inputName);
        }
        /// <summary>
        /// Calculates the output model values for each of the input examples. (Inference).
        /// </summary>
        /// <typeparam name="T">Data type. Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="data">A set of examples for which the output is calculated</param>
        /// <param name="device">Device for calculations</param>
        /// <param name="inputName">The name of the input layer. The name must be unique throughout the network. There can be several inputs, this parameter indicates which of them serves data.</param>
        /// <returns></returns>
        public static IEnumerable<T[][]> Predict<T>(this SequentialMultiOutput<T> source,
           IEnumerable<Value> data,
           DeviceDescriptor device,
           string inputName = "Input") where T : IConvertible
        {
            return source.Model.PredictMultiOutput<T>(data, device, inputName);
        }
        /// <summary>
        /// Calculates the output model values for each of the input examples. (Inference).
        /// </summary>
        /// <typeparam name="T">Data type. Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="data">A set of examples for which the output is calculated</param>
        /// <param name="device">Device for calculations</param>
        /// <param name="inputName">The name of the input layer. The name must be unique throughout the network. There can be several inputs, this parameter indicates which of them serves data.</param>
        public static IEnumerable<T[][]> Predict<T>(this SequentialMultiOutput<T> source,
            IEnumerable<T[]> data,
            DeviceDescriptor device,
            int minibatchSize = 512,
            string inputName = "Input") where T : IConvertible
        {
            return source.Model.PredictMultiOutput<T>(data, device, minibatchSize, inputName);
        }
        /// <summary>
        /// Calculates the output model values for each of the input examples (example is a sequence). (Inference).
        /// </summary>
        /// <typeparam name="T">Data type. Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="device">Device for calculations</param>
        /// <param name="inputName">The name of the input layer. The name must be unique throughout the network. There can be several inputs, this parameter indicates which of them serves data.</param>
        /// <returns></returns>
        public static IEnumerable<T[][]> Predict<T>(this SequentialMultiOutput<T> source,
            IEnumerable<IList<T[]>> data,
            DeviceDescriptor device,
            int minibatchSize = 512,
            string inputName = "Input") where T : IConvertible
        {
            return source.Model.PredictMultiOutput<T>(data, device, minibatchSize, inputName);
        }
        /// <summary>
        /// Calculates the output model values for each of the input examples (example - 2D). (Inference).
        /// </summary>
        /// <typeparam name="T">Data type. Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="device">Device for calculations</param>
        /// <param name="inputName">The name of the input layer. The name must be unique throughout the network. There can be several inputs, this parameter indicates which of them serves data.</param>
        /// <returns></returns>
        public static IEnumerable<T[][]> Predict<T>(this SequentialMultiOutput<T> source,
            IEnumerable<T[,]> data,
            DeviceDescriptor device,
            int minibatchSize = 512,
            string inputName = "Input") where T : IConvertible
        {
            return source.Model.PredictMultiOutput<T>(data, device, minibatchSize, inputName);
        }
        /// <summary>
        /// Computes the output model values of one example. (Inference).
        /// </summary>
        /// <typeparam name="T">Data type. Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="device">Device for calculations</param>
        /// <param name="inputName">The name of the input layer. The name must be unique throughout the network. There can be several inputs, this parameter indicates which of them serves data.</param>
        /// <returns></returns>
        public static T[][] Predict<T>(this SequentialMultiOutput<T> source, T[] data, DeviceDescriptor device, string inputName = "Input") where T : IConvertible
        {
            return source.Model.PredictMultiOutput<T>(data, device, inputName);
        }
        /// <summary>
        /// Computes the output values of the model of one example (example - sequence). (Inference).
        /// </summary>
        /// <typeparam name="T">Data type. Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="device">Device for calculations</param>
        /// <param name="inputName">The name of the input layer. The name must be unique throughout the network. There can be several inputs, this parameter indicates which of them serves data.</param>
        /// <returns></returns>
        public static T[][] Predict<T>(this SequentialMultiOutput<T> source, IList<T[]> data, DeviceDescriptor device, string inputName = "Input") where T : IConvertible
        {
            return source.Model.PredictMultiOutput<T>(data, device, inputName);
        }
        /// <summary>
        /// Computes the output values of the model of one example (example - 2D). (Inference).
        /// </summary>
        /// <typeparam name="T">Data type. Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="device">Device for calculations</param>
        /// <param name="inputName">The name of the input layer. The name must be unique throughout the network. There can be several inputs, this parameter indicates which of them serves data.</param>
        /// <returns></returns>
        public static T[][] Predict<T>(this SequentialMultiOutput<T> source, T[,] data, DeviceDescriptor device, string inputName = "Input") where T : IConvertible
        {
            return source.Model.PredictMultiOutput<T>(data, device, inputName);
        } 
        #endregion

        #endregion

        #endregion
    }
}
