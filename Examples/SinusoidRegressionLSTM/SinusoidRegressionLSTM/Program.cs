using CNTK;
using System;
using System.Collections.Generic;
using System.Linq;
using EasyCNTK;
using EasyCNTK.ActivationFunctions;
using EasyCNTK.Layers;
using EasyCNTK.Learning;
using EasyCNTK.Learning.Metrics;
using EasyCNTK.Learning.Optimizers;
using EasyCNTK.LossFunctions;

namespace SinusoidRegressionLSTM
{
    class Program
    {
        static void Main(string[] args)
        {
            CNTKLib.SetFixedRandomSeed(0); //for reproducibility. because initialization of weights in neural network layers
                                           //depends on CNTK random number generator

            //create a simulated dataset from sequences describing a sinusoid
            var dataset = Enumerable.Range(1, 2000)
                .Select(p => Math.Sin(p / 100.0)) //decrease the pitch so that the sine wave is smoother
                .Segment(10) //break the sinusoid into segments of 10 elements
                .Select(p => (featureSequence: p.Take(9).Select(q => new[] { q }).ToArray(), //set a sequence of 9 elements, each element of dimension 1 (maybe: 1, 2, 3 ... n)
                                        label: new[] { p[9] })) //set a label for a sequence of dimension 1 (maybe: 1, 2, 3 ... n)
                .ToArray();
            dataset.Split(0.7, out var train, out var test);

            int minibatchSize = 16;
            int epochCount = 300;
            int inputDimension = 1;
            var device = DeviceDescriptor.GPUDevice(0);

            var model = new Sequential<double>(device, new[] { inputDimension }, inputName: "Input");
            model.Add(new LSTM(1, selfStabilizerLayer: new SelfStabilization()));
            model.Add(new Residual2(1, new Tanh()));

            //it is possible to join LSTM layers one after another as in the comment below:
            //var model = new Sequential<double>(device, new[] { inputDimension });
            //model.Add(new Dense(3, new Tanh())); 
            //model.Add (new LSTM (10, isLastLstm: false)); // LSTM can also be the first layer in the model
            //model.Add(new LSTM(5, isLastLstm: false));
            //model.Add(new LSTM(2, selfStabilizerLayer: new SelfStabilization())); 
            //model.Add(new Residual2(1, new Tanh()));

            //uses one of several overloads that can train recursive networks
            var fitResult = model.Fit(features:     train.Select(p => p.featureSequence).ToArray(), 
                labels:                             train.Select(p => p.label).ToArray(),
                minibatchSize:                      minibatchSize,
                lossFunction:                       new AbsoluteError(),
                evaluationFunction:                 new AbsoluteError(),
                optimizer:                          new Adam(0.005, 0.9, minibatchSize),
                epochCount:                         epochCount,
                device:                             device,
                shuffleSampleInMinibatchesPerEpoch: true,
                ruleUpdateLearningRate: (epoch, learningRate) => learningRate % 50 == 0 ? 0.95 * learningRate : learningRate,
                actionPerEpoch: (epoch, loss, eval) =>
                {
                    Console.WriteLine($"Loss: {loss:F10} Eval: {eval:F3} Epoch: {epoch}");
                    if (loss < 0.05) //stopping criterion is reached, save the model to a file and finish training (approximately 112 epochs)
                    {
                        model.SaveModel($"{model}.model", saveArchitectureDescription: false);
                        return true;
                    }
                    return false;
                }, 
                inputName: "Input");

            Console.WriteLine($"Duration train: {fitResult.Duration}");
            Console.WriteLine($"Epochs: {fitResult.EpochCount}");
            Console.WriteLine($"Loss error: {fitResult.LossError}");
            Console.WriteLine($"Eval error: {fitResult.EvaluationError}");

            var metricsTrain = model
                .Evaluate(train.Select(p => p.featureSequence), train.Select(p => p.label), device)
                .GetRegressionMetrics();
            var metricsTest = model
                .Evaluate(test.Select(p => p.featureSequence), test.Select(p => p.label), device)
                .GetRegressionMetrics();
            
            Console.WriteLine($"Train => MAE: {metricsTrain[0].MAE} RMSE: {metricsTrain[0].RMSE} R2: {metricsTrain[0].Determination}");//R2 ~ 0,983
            Console.WriteLine($"Test => MAE: {metricsTest[0].MAE} RMSE: {metricsTest[0].RMSE} R2: {metricsTest[0].Determination}"); //R2 ~ 0,982

            Console.ReadKey();
        }
    }
}
