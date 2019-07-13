﻿//
// Copyright (c) Stanislav Grigoriev. All rights reserved.
// grigorievstas9@gmail.com 
// https://github.com/StanislavGrigoriev/EasyCNTK
//
// Copyright (c) Microsoft. All rights reserved.
//
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//
using CNTK;
using EasyCNTK.Learning.Optimizers;
using EasyCNTK.LossFunctions;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace EasyCNTK.Learning
{
    public static class LearnExtensions
    {
        #region Extensions for Function

        public static FitResult Fit(this Trainer trainer,
            DeviceDescriptor device,
            MinibatchSource miniBatchSource, uint miniBatchSize,
            Dictionary<Variable, StreamInformation> streamInfos,
            Learner learner,
            double learningRate,
            int epochs,
            Func<int, double, double> ruleUpdateLearningRate = null,
            Func<FitResult, bool> actionPerEpoch = null,
            IProgress<double> progress = null)
        {
            var losses = new List<double>();
            var evals = new List<double>();

            var sw = new Stopwatch();
            sw.Start();

            var epochCount = 1;
            var failureCount = 0;
            while (epochCount <= epochs)
            {
                if (!TrainSingleEpoch(trainer, device, miniBatchSource, miniBatchSize, streamInfos) && failureCount < 3)
                {
                    failureCount++;
                    Console.WriteLine($"Attempt: {failureCount + 1} to train Epoch: {epochCount}");
                    continue;
                }


                losses.Add(trainer.PreviousMinibatchLossAverage());
                evals.Add(trainer.PreviousMinibatchEvaluationAverage());

                if (actionPerEpoch != null)
                {
                    var result = new FitResult()
                    {
                        Duration = sw.Elapsed,
                        EpochCount = epochCount,
                        EvaluationCurve = evals,
                        EvaluationError = evals.Last(),
                        LossCurve = losses,
                        LossError = losses.Last()
                    };

                    var stopTraining = actionPerEpoch(result);

                    if (stopTraining)
                    {
                        progress?.Report(1);
                        break;
                    }
                }

                if (ruleUpdateLearningRate != null)
                    learningRate = UpdateLearningRate(ruleUpdateLearningRate, epochCount, learningRate, learner);



                progress?.Report(1d * epochCount / epochs);

                epochCount++;
            }

            sw.Stop();

            return new FitResult()
            {
                LossError = losses[losses.Count - 1],
                EvaluationError = evals[evals.Count - 1],
                Duration = sw.Elapsed,
                EpochCount = epochCount,
                LossCurve = losses,
                EvaluationCurve = evals
            };
        }

        private static bool TrainSingleEpoch(Trainer trainer, DeviceDescriptor device,
            MinibatchSource miniBatchSource, uint miniBatchSize,
            Dictionary<Variable, StreamInformation> streamInfos)
        {

            try
            {
                var miniBatchData = miniBatchSource.GetNextMinibatch(miniBatchSize, device);

                while (!MiniBatchDataIsSweepEnd(miniBatchData.Values))
                {
                    var arguments = streamInfos.ToDictionary(kv => kv.Key, kv => miniBatchData[kv.Value]);

                    trainer.TrainMinibatch(arguments, device);

                    miniBatchData = miniBatchSource.GetNextMinibatch(miniBatchSize, device);
                }

                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex);
                return false;
            }
        }

        private static double UpdateLearningRate(Func<int, double, double> ruleUpdateLearningRate, int epochCount, double learningRate, Learner learner)
        {
            var proposedLearningRate = ruleUpdateLearningRate(epochCount, learningRate);

            if (!(Math.Abs(proposedLearningRate - learningRate) > double.Epsilon)) return learningRate;

            learningRate = proposedLearningRate;
            learner.SetLearningRateSchedule(new TrainingParameterScheduleDouble(learningRate));
            return learningRate;
        }

        public static bool MiniBatchDataIsSweepEnd(ICollection<MinibatchData> miniBatchValues) => miniBatchValues.Any(a => a.sweepEnd);


        /// <summary>
        /// Обучает модель. Поддерживает реккурентные сети.
        /// </summary>
        /// <param name="source"></param>
        /// <param name="trainDataSelector">Селектор, позволяющий указать для каждой эпохи свой набор данных для обучения.</param>
        /// <param name="lossFunction">Функция потерь</param>
        /// <param name="evaluationFunction">Оценочная функция</param>
        /// <param name="optimizer">Оптимизатор, используемый для обучения</param>
        /// <param name="epochCount">Количество эпох обучения</param>
        /// <param name="isReccurentModel">Указывает, что требуется обучать реккурентную модель</param>
        /// <param name="device">Устройство для обучения</param>
        /// <param name="ruleUpdateLearningRate">Правило обновления скорости обучения. Входные параметры: эпоха, текущая скорость обучения. Выходные: новая скорость обучения.</param>
        /// <param name="actionPerEpoch">Произвольное действие, которое требуется выполнять каждую эпоху. Позволяет прервать процесс тренировки. Входные параметры: эпоха, loss-ошибка, evaluation-ошибка. 
        /// Выходные: true - прервать процесс тренировки, false - продолжить тренировку.
        /// Используется для осуществления логирования, отображения процесса обучения, сохранения промежуточных чекпоинтов модели и т.п.</param>
        /// <returns></returns>
        public static FitResult Fit(this Function source,
           Func<int, IEnumerable<Minibatch>> trainDataSelector,
           Loss lossFunction,
           Loss evaluationFunction,
           Optimizer optimizer,
           int epochCount,
           bool isReccurentModel,
           DeviceDescriptor device,
           Func<int, double, double> ruleUpdateLearningRate = null,
           Func<int, double, double, bool> actionPerEpoch = null)
        {
            var inputVariable = source.Inputs.Single(p => p.Name.ToUpper() == "INPUT");
            var outputVariable = isReccurentModel
                ? Variable.InputVariable(source.Output.Shape, source.Output.DataType, "output", new List<Axis> { Axis.DefaultBatchAxis() })
                : Variable.InputVariable(source.Output.Shape, source.Output.DataType, "output");

            var loss = lossFunction.GetLoss(source, outputVariable, device);
            var evaluation = evaluationFunction.GetLoss(source, outputVariable, device);
            var learner = optimizer.GetOptimizer(source.Parameters());
            var trainer = CNTKLib.CreateTrainer(
                source,
                loss,
                evaluation,
                new LearnerVector() { learner });
            var learningRate = optimizer.LearningRate;
            var losses = new List<double>(epochCount);
            var evals = new List<double>(epochCount);
            Stopwatch sw = new Stopwatch();
            sw.Start();
            for (int i = 1; i <= epochCount; i++)
            {
                var trainMinibatches = trainDataSelector(i);
                foreach (var miniBatch in trainMinibatches)
                {
                    trainer.TrainMinibatch(new Dictionary<Variable, Value>() { { inputVariable, miniBatch.Features }, { outputVariable, miniBatch.Labels } }, false, device);
                }
                losses.Add(trainer.PreviousMinibatchLossAverage());
                evals.Add(trainer.PreviousMinibatchEvaluationAverage());

                bool needStopTraining = actionPerEpoch?.Invoke(i, losses[i - 1], evals[i - 1]) ?? false;
                if (needStopTraining)
                {
                    epochCount = i;
                    break;
                }

                if (ruleUpdateLearningRate != null)
                {
                    var proposaledLearningRate = ruleUpdateLearningRate(i, learningRate);
                    if (proposaledLearningRate != learningRate)
                    {
                        learner.SetLearningRateSchedule(new TrainingParameterScheduleDouble(learningRate));
                        learningRate = proposaledLearningRate;
                    }
                }
            }
            sw.Stop();

            return new FitResult()
            {
                LossError = losses[losses.Count - 1],
                EvaluationError = evals[evals.Count - 1],
                Duration = sw.Elapsed,
                EpochCount = epochCount,
                LossCurve = losses,
                EvaluationCurve = evals
            };
        }

        /// <summary>
        /// Обучает модель. Поддерживает реккурентные сети.
        /// </summary>
        /// <param name="source"></param>
        /// <param name="trainData">Набор данных для обучения.</param>
        /// <param name="lossFunction">Функция потерь</param>
        /// <param name="evaluationFunction">Оценочная функция</param>
        /// <param name="optimizer">Оптимизатор, используемый для обучения</param>
        /// <param name="epochCount">Количество эпох обучения</param>
        /// <param name="isReccurentModel">Указывает, что требуется обучать реккурентную модель</param>
        /// <param name="device">Устройство для обучения</param>
        /// <param name="ruleUpdateLearningRate">Правило обновления скорости обучения. Входные параметры: эпоха, текущая скорость обучения. Выходные: новая скорость обучения.</param>
        /// <param name="actionPerEpoch">Произвольное действие, которое требуется выполнять каждую эпоху. Позволяет прервать процесс тренировки. Входные параметры: эпоха, loss-ошибка, evaluation-ошибка. 
        /// Выходные: true - прервать процесс тренировки, false - продолжить тренировку.
        /// Используется для осуществления логирования, отображения процесса обучения, сохранения промежуточных чекпоинтов модели и т.п.</param>
        /// <returns></returns>
        public static FitResult Fit(this Function source,
            IEnumerable<Minibatch> trainData,
            Loss lossFunction,
            Loss evaluationFunction,
            Optimizer optimizer,
            int epochCount,
            DeviceDescriptor device,
            bool isReccurentModel = false,
            Func<int, double, double> ruleUpdateLearningRate = null,
            Func<int, double, double, bool> actionPerEpoch = null)
        {
            return source.Fit(p => trainData, lossFunction, evaluationFunction, optimizer, epochCount, isReccurentModel, device, ruleUpdateLearningRate, actionPerEpoch);
        }
        /// <summary>
        /// Обучает модель. Не применим для обучения реккуретных сетей.
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="trainData">Набор данных для обучения. Каждый пример должен содержать в начале массива признаки размерностью inputDim, а в конце метки классов размерностью outputDim.
        /// Например inputDim = 3, outputDim = 2: [f1, f2, f3, l1, l2]</param>
        /// <param name="inputDim">Размерность признаков (разрядность)</param>
        /// <param name="minibatchSize">Размер минипакета</param>
        /// <param name="lossFunction">Функция потерь</param>
        /// <param name="evaluationFunction">Оценочная функция</param>
        /// <param name="optimizer">Оптимизатор, используемый для обучения</param>
        /// <param name="epochCount">Количество эпох обучения</param>        
        /// <param name="device">Устройство для обучения</param>
        /// <param name="shuffleSampleInMinibatchesPerEpoch">Указывает, что необходимо каждую эпоху перемешивать обучающие примеры для формирования новых минипакетов.</param>
        /// <param name="ruleUpdateLearningRate">Правило обновления скорости обучения. Входные параметры: эпоха, текущая скорость обучения. Выходные: новая скорость обучения.</param>
        /// <param name="actionPerEpoch">Произвольное действие, которое требуется выполнять каждую эпоху. Позволяет прервать процесс тренировки. Входные параметры: эпоха, loss-ошибка, evaluation-ошибка. 
        /// Выходные: true - прервать процесс тренировки, false - продолжить тренировку.
        /// Используется для осуществления логирования, отображения процесса обучения, сохранения промежуточных чекпоинтов модели и т.п.</param>
        /// <returns></returns>
        public static FitResult Fit<T>(this Function source,
            IList<T[]> trainData,
            int inputDim,
            int minibatchSize,
            Loss lossFunction,
            Loss evaluationFunction,
            Optimizer optimizer,
            int epochCount,
            DeviceDescriptor device,
            bool shuffleSampleInMinibatchesPerEpoch,
            Func<int, double, double> ruleUpdateLearningRate = null,
            Func<int, double, double, bool> actionPerEpoch = null) where T : IConvertible
        {
            DataConverter valueConverter = new DataConverter();
            IList<Minibatch> minibatches = null;
            if (!shuffleSampleInMinibatchesPerEpoch)
            {
                minibatches = valueConverter.ConvertDatasetToMinibatch(trainData, inputDim, minibatchSize, device).ToArray();
            }
            Func<int, IEnumerable<Minibatch>> getMinibatches = epoch =>
            {
                if (shuffleSampleInMinibatchesPerEpoch)
                {
                    trainData.Shuffle();
                    minibatches = valueConverter.ConvertDatasetToMinibatch(trainData, inputDim, minibatchSize, device).ToArray();
                }
                return minibatches;
            };
            return source.Fit(getMinibatches, lossFunction, evaluationFunction, optimizer, epochCount, false, device, ruleUpdateLearningRate, actionPerEpoch);
        }
        /// <summary>
        /// Обучает реккурентную модель.
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="features">Набор последовательностей (признаков). Каждая последовательность может быть переменной длинны, но одинаковой размерности (массивы из которых состоит последовательность, должны иметь одинаковую длину)</param>
        /// <param name="labels">Набор меток. Размерность меток должна быть одинаковая.</param>
        /// <param name="minibatchSize">Размер минипакета</param>
        /// <param name="lossFunction">Функция потерь</param>
        /// <param name="evaluationFunction">Оценочная функция</param>
        /// <param name="optimizer">Оптимизатор, используемый для обучения</param>
        /// <param name="epochCount">Количество эпох обучения</param>        
        /// <param name="device">Устройство для обучения</param>
        /// <param name="shuffleSampleInMinibatchesPerEpoch">Указывает, что необходимо каждую эпоху перемешивать обучающие примеры для формирования новых минипакетов.</param>
        /// <param name="ruleUpdateLearningRate">Правило обновления скорости обучения. Входные параметры: эпоха, текущая скорость обучения. Выходные: новая скорость обучения.</param>
        /// <param name="actionPerEpoch">Произвольное действие, которое требуется выполнять каждую эпоху. Позволяет прервать процесс тренировки. Входные параметры: эпоха, loss-ошибка, evaluation-ошибка. 
        /// Выходные: true - прервать процесс тренировки, false - продолжить тренировку.
        /// Используется для осуществления логирования, отображения процесса обучения, сохранения промежуточных чекпоинтов модели и т.п.</param>
        /// <returns></returns>
        public static FitResult Fit<T>(this Function source,
            IList<IList<T[]>> features,
            IList<T[]> labels,
            int minibatchSize,
            Loss lossFunction,
            Loss evaluationFunction,
            Optimizer optimizer,
            int epochCount,
            DeviceDescriptor device,
            bool shuffleSampleInMinibatchesPerEpoch,
            Func<int, double, double> ruleUpdateLearningRate = null,
            Func<int, double, double, bool> actionPerEpoch = null) where T : IConvertible
        {
            if (features.Count != labels.Count) throw new ArgumentException("Количество поледовательностей(features) и меток(labels) должно быть одинаковым.");

            DataConverter valueConverter = new DataConverter();
            IList<Minibatch> minibatches = null;
            if (!shuffleSampleInMinibatchesPerEpoch)
            {
                minibatches = valueConverter.ConvertDatasetToMinibatch(features, labels, minibatchSize, device).ToArray();
            }
            var trainData = features.Zip(labels, (f, l) => (f, l)).ToArray();
            Func<int, IEnumerable<Minibatch>> getMinibatches = epoch =>
            {
                if (shuffleSampleInMinibatchesPerEpoch)
                {
                    trainData.Shuffle();
                    minibatches = valueConverter.ConvertDatasetToMinibatch(trainData.Select(p => p.f), trainData.Select(p => p.l), minibatchSize, device).ToArray();
                }
                return minibatches;
            };

            return source.Fit(getMinibatches, lossFunction, evaluationFunction, optimizer, epochCount, true, device, ruleUpdateLearningRate, actionPerEpoch);
        }
        /// <summary>
        /// Обучает модель с двумерным входом. Не применим для обучения реккуретных сетей.
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="trainData">Набор данных для обучения.</param>
        /// <param name="minibatchSize">Размер минипакета</param>
        /// <param name="lossFunction">Функция потерь</param>
        /// <param name="evaluationFunction">Оценочная функция</param>
        /// <param name="optimizer">Оптимизатор, используемый для обучения</param>
        /// <param name="epochCount">Количество эпох обучения</param>        
        /// <param name="device">Устройство для обучения</param>
        /// <param name="shuffleSampleInMinibatchesPerEpoch">Указывает, что необходимо каждую эпоху перемешивать обучающие примеры для формирования новых минипакетов.</param>
        /// <param name="ruleUpdateLearningRate">Правило обновления скорости обучения. Входные параметры: эпоха, текущая скорость обучения. Выходные: новая скорость обучения.</param>
        /// <param name="actionPerEpoch">Произвольное действие, которое требуется выполнять каждую эпоху. Позволяет прервать процесс тренировки. Входные параметры: эпоха, loss-ошибка, evaluation-ошибка. 
        /// Выходные: true - прервать процесс тренировки, false - продолжить тренировку.
        /// Используется для осуществления логирования, отображения процесса обучения, сохранения промежуточных чекпоинтов модели и т.п.</param>
        /// <returns></returns>
        public static FitResult Fit<T>(this Function source,
            IList<Sample2D<T>> trainData,
            int minibatchSize,
            Loss lossFunction,
            Loss evaluationFunction,
            Optimizer optimizer,
            int epochCount,
            DeviceDescriptor device,
            bool shuffleSampleInMinibatchesPerEpoch,
            Func<int, double, double> ruleUpdateLearningRate = null,
            Func<int, double, double, bool> actionPerEpoch = null) where T : IConvertible
        {
            DataConverter valueConverter = new DataConverter();
            IList<Minibatch> minibatches = null;
            if (!shuffleSampleInMinibatchesPerEpoch)
            {
                minibatches = valueConverter.ConvertDatasetToMinibatch(trainData, minibatchSize, device).ToArray();
            }
            Func<int, IEnumerable<Minibatch>> getMinibatches = epoch =>
            {
                if (shuffleSampleInMinibatchesPerEpoch)
                {
                    trainData.Shuffle();
                    minibatches = valueConverter.ConvertDatasetToMinibatch(trainData, minibatchSize, device).ToArray();
                }
                return minibatches;
            };
            return source.Fit(getMinibatches, lossFunction, evaluationFunction, optimizer, epochCount, false, device, ruleUpdateLearningRate, actionPerEpoch);
        }
        /// <summary>
        /// Обучает модель. Не применим для обучения реккуретных сетей.
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="trainData">Набор данных для обучения. Каждый пример должен содержать в начале массива признаки размерностью inputDim, а в конце метки классов размерностью outputDim.
        /// Например inputDim = 3, outputDim = 2: [f1, f2, f3, l1, l2]</param>
        /// <param name="inputDim">Размерность признаков (разрядность)</param>
        /// <param name="minibatchSize">Размер минипакета</param>
        /// <param name="lossFunction">Функция потерь</param>
        /// <param name="evaluationFunction">Оценочная функция</param>
        /// <param name="optimizer">Оптимизатор, используемый для обучения</param>
        /// <param name="epochCount">Количество эпох обучения</param>        
        /// <param name="device">Устройство для обучения</param>       
        /// <param name="ruleUpdateLearningRate">Правило обновления скорости обучения. Входные параметры: эпоха, текущая скорость обучения. Выходные: новая скорость обучения.</param>
        /// <param name="actionPerEpoch">Произвольное действие, которое требуется выполнять каждую эпоху. Позволяет прервать процесс тренировки. Входные параметры: эпоха, loss-ошибка, evaluation-ошибка. 
        /// Выходные: true - прервать процесс тренировки, false - продолжить тренировку.
        /// Используется для осуществления логирования, отображения процесса обучения, сохранения промежуточных чекпоинтов модели и т.п.</param>
        /// <returns></returns>
        public static FitResult Fit<T>(this Function source,
            IEnumerable<T[]> trainData,
            int inputDim,
            int minibatchSize,
            Loss lossFunction,
            Loss evaluationFunction,
            Optimizer optimizer,
            int epochCount,
            DeviceDescriptor device,
            Func<int, double, double> ruleUpdateLearningRate = null,
            Func<int, double, double, bool> actionPerEpoch = null) where T : IConvertible
        {
            DataConverter valueConverter = new DataConverter();
            var minibatches = valueConverter.ConvertDatasetToMinibatch(trainData, inputDim, minibatchSize, device);
            return source.Fit(p => minibatches, lossFunction, evaluationFunction, optimizer, epochCount, false, device, ruleUpdateLearningRate, actionPerEpoch);
        }
        /// <summary>
        /// Обучает реккурентную модель.
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="features">Набор последовательностей (признаков). Каждая последовательность может быть переменной длинны, но одинаковой размерности (массивы из которых состоит последовательность, должны иметь одинаковую длину)</param>
        /// <param name="labels">Набор меток. Размерность меток должна быть одинаковая.</param>
        /// <param name="minibatchSize">Размер минипакета</param>
        /// <param name="lossFunction">Функция потерь</param>
        /// <param name="evaluationFunction">Оценочная функция</param>
        /// <param name="optimizer">Оптимизатор, используемый для обучения</param>
        /// <param name="epochCount">Количество эпох обучения</param>        
        /// <param name="device">Устройство для обучения</param>     
        /// <param name="ruleUpdateLearningRate">Правило обновления скорости обучения. Входные параметры: эпоха, текущая скорость обучения. Выходные: новая скорость обучения.</param>
        /// <param name="actionPerEpoch">Произвольное действие, которое требуется выполнять каждую эпоху. Позволяет прервать процесс тренировки. Входные параметры: эпоха, loss-ошибка, evaluation-ошибка. 
        /// Выходные: true - прервать процесс тренировки, false - продолжить тренировку.
        /// Используется для осуществления логирования, отображения процесса обучения, сохранения промежуточных чекпоинтов модели и т.п.</param>
        /// <returns></returns>
        public static FitResult Fit<T>(this Function source,
            IEnumerable<IList<T[]>> features,
            IEnumerable<T[]> labels,
            int minibatchSize,
            Loss lossFunction,
            Loss evaluationFunction,
            Optimizer optimizer,
            int epochCount,
            DeviceDescriptor device,
            Func<int, double, double> ruleUpdateLearningRate = null,
            Func<int, double, double, bool> actionPerEpoch = null) where T : IConvertible
        {
            DataConverter valueConverter = new DataConverter();
            var minibatches = valueConverter.ConvertDatasetToMinibatch(features, labels, minibatchSize, device);
            return source.Fit(p => minibatches, lossFunction, evaluationFunction, optimizer, epochCount, true, device, ruleUpdateLearningRate, actionPerEpoch);
        }
        /// <summary>
        /// Обучает модель с двумерным входом. Не применим для обучения реккуретных сетей.
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="trainData">Набор данных для обучения.</param>
        /// <param name="minibatchSize">Размер минипакета</param>
        /// <param name="lossFunction">Функция потерь</param>
        /// <param name="evaluationFunction">Оценочная функция</param>
        /// <param name="optimizer">Оптимизатор, используемый для обучения</param>
        /// <param name="epochCount">Количество эпох обучения</param>        
        /// <param name="device">Устройство для обучения</param>  
        /// <param name="ruleUpdateLearningRate">Правило обновления скорости обучения. Входные параметры: эпоха, текущая скорость обучения. Выходные: новая скорость обучения.</param>
        /// <param name="actionPerEpoch">Произвольное действие, которое требуется выполнять каждую эпоху. Позволяет прервать процесс тренировки. Входные параметры: эпоха, loss-ошибка, evaluation-ошибка. 
        /// Выходные: true - прервать процесс тренировки, false - продолжить тренировку.
        /// Используется для осуществления логирования, отображения процесса обучения, сохранения промежуточных чекпоинтов модели и т.п.</param>
        /// <returns></returns>
        public static FitResult Fit<T>(this Function source,
            IEnumerable<Sample2D<T>> trainData,
            int minibatchSize,
            Loss lossFunction,
            Loss evaluationFunction,
            Optimizer optimizer,
            int epochCount,
            DeviceDescriptor device,
            Func<int, double, double> ruleUpdateLearningRate = null,
            Func<int, double, double, bool> actionPerEpoch = null) where T : IConvertible
        {
            DataConverter valueConverter = new DataConverter();
            var minibatches = valueConverter.ConvertDatasetToMinibatch(trainData, minibatchSize, device);
            return source.Fit(p => minibatches, lossFunction, evaluationFunction, optimizer, epochCount, false, device, ruleUpdateLearningRate, actionPerEpoch);
        }
        #endregion

        #region Extensions for Sequential<T>
        /// <summary>
        /// Обучает модель. Поддерживает реккурентные сети.
        /// </summary>
        /// <param name="source"></param>
        /// <param name="trainDataSelector">Селектор, позволяющий указать для каждой эпохи свой набор данных для обучения.</param>
        /// <param name="lossFunction">Функция потерь</param>
        /// <param name="evaluationFunction">Оценочная функция</param>
        /// <param name="optimizer">Оптимизатор, используемый для обучения</param>
        /// <param name="epochCount">Количество эпох обучения</param>
        /// <param name="isReccurentModel">Указывает, что требуется обучать реккурентную модель</param>
        /// <param name="device">Устройство для обучения</param>
        /// <param name="ruleUpdateLearningRate">Правило обновления скорости обучения. Входные параметры: эпоха, текущая скорость обучения. Выходные: новая скорость обучения.</param>
        /// <param name="actionPerEpoch">Произвольное действие, которое требуется выполнять каждую эпоху. Позволяет прервать процесс тренировки. Входные параметры: эпоха, loss-ошибка, evaluation-ошибка. 
        /// Выходные: true - прервать процесс тренировки, false - продолжить тренировку.
        /// Используется для осуществления логирования, отображения процесса обучения, сохранения промежуточных чекпоинтов модели и т.п.</param>
        /// <returns></returns>
        public static FitResult Fit<T>(this Sequential<T> source,
           Func<int, IEnumerable<Minibatch>> trainDataSelector,
           Loss lossFunction,
           Loss evaluationFunction,
           Optimizer optimizer,
           int epochCount,
           bool isReccurentModel,
           DeviceDescriptor device,
           Func<int, double, double> ruleUpdateLearningRate = null,
           Func<int, double, double, bool> actionPerEpoch = null) where T : IConvertible
        {
            return source.Model.Fit(trainDataSelector, lossFunction, evaluationFunction, optimizer, epochCount, isReccurentModel, device, ruleUpdateLearningRate, actionPerEpoch);
        }
        /// <summary>
        /// Обучает модель. Поддерживает реккурентные сети.
        /// </summary>
        /// <param name="source"></param>
        /// <param name="trainData">Набор данных для обучения.</param>
        /// <param name="lossFunction">Функция потерь</param>
        /// <param name="evaluationFunction">Оценочная функция</param>
        /// <param name="optimizer">Оптимизатор, используемый для обучения</param>
        /// <param name="epochCount">Количество эпох обучения</param>
        /// <param name="isReccurentModel">Указывает, что требуется обучать реккурентную модель</param>
        /// <param name="device">Устройство для обучения</param>
        /// <param name="ruleUpdateLearningRate">Правило обновления скорости обучения. Входные параметры: эпоха, текущая скорость обучения. Выходные: новая скорость обучения.</param>
        /// <param name="actionPerEpoch">Произвольное действие, которое требуется выполнять каждую эпоху. Позволяет прервать процесс тренировки. Входные параметры: эпоха, loss-ошибка, evaluation-ошибка. 
        /// Выходные: true - прервать процесс тренировки, false - продолжить тренировку.
        /// Используется для осуществления логирования, отображения процесса обучения, сохранения промежуточных чекпоинтов модели и т.п.</param>
        /// <returns></returns>
        public static FitResult Fit<T>(this Sequential<T> source,
            IEnumerable<Minibatch> trainData,
            Loss lossFunction,
            Loss evaluationFunction,
            Optimizer optimizer,
            int epochCount,
            DeviceDescriptor device,
            bool isReccurentModel = false,
            Func<int, double, double> ruleUpdateLearningRate = null,
            Func<int, double, double, bool> actionPerEpoch = null) where T : IConvertible
        {
            return source.Model.Fit(p => trainData, lossFunction, evaluationFunction, optimizer, epochCount, isReccurentModel, device, ruleUpdateLearningRate, actionPerEpoch);
        }
        /// <summary>
        /// Обучает модель. Не применим для обучения реккуретных сетей.
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="trainData">Набор данных для обучения. Каждый пример должен содержать в начале массива признаки размерностью inputDim, а в конце метки классов размерностью outputDim.
        /// Например inputDim = 3, outputDim = 2: [f1, f2, f3, l1, l2]</param>
        /// <param name="inputDim">Размерность признаков (разрядность)</param>
        /// <param name="minibatchSize">Размер минипакета</param>
        /// <param name="lossFunction">Функция потерь</param>
        /// <param name="evaluationFunction">Оценочная функция</param>
        /// <param name="optimizer">Оптимизатор, используемый для обучения</param>
        /// <param name="epochCount">Количество эпох обучения</param>        
        /// <param name="device">Устройство для обучения</param>
        /// <param name="shuffleSampleInMinibatchesPerEpoch">Указывает, что необходимо каждую эпоху перемешивать обучающие примеры для формирования новых минипакетов.</param>
        /// <param name="ruleUpdateLearningRate">Правило обновления скорости обучения. Входные параметры: эпоха, текущая скорость обучения. Выходные: новая скорость обучения.</param>
        /// <param name="actionPerEpoch">Произвольное действие, которое требуется выполнять каждую эпоху. Позволяет прервать процесс тренировки. Входные параметры: эпоха, loss-ошибка, evaluation-ошибка. 
        /// Выходные: true - прервать процесс тренировки, false - продолжить тренировку.
        /// Используется для осуществления логирования, отображения процесса обучения, сохранения промежуточных чекпоинтов модели и т.п.</param>
        /// <returns></returns>
        public static FitResult Fit<T>(this Sequential<T> source,
            IList<T[]> trainData,
            int inputDim,
            int minibatchSize,
            Loss lossFunction,
            Loss evaluationFunction,
            Optimizer optimizer,
            int epochCount,
            DeviceDescriptor device,
            bool shuffleSampleInMinibatchesPerEpoch,
            Func<int, double, double> ruleUpdateLearningRate = null,
            Func<int, double, double, bool> actionPerEpoch = null) where T : IConvertible
        {
            return source.Model.Fit(trainData, inputDim, minibatchSize, lossFunction, evaluationFunction, optimizer, epochCount, device, shuffleSampleInMinibatchesPerEpoch, ruleUpdateLearningRate, actionPerEpoch);
        }
        /// <summary>
        /// Обучает реккурентную модель.
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="features">Набор последовательностей (признаков). Каждая последовательность может быть переменной длинны, но одинаковой размерности (массивы из которых состоит последовательность, должны иметь одинаковую длину)</param>
        /// <param name="labels">Набор меток. Размерность меток должна быть одинаковая.</param>
        /// <param name="minibatchSize">Размер минипакета</param>
        /// <param name="lossFunction">Функция потерь</param>
        /// <param name="evaluationFunction">Оценочная функция</param>
        /// <param name="optimizer">Оптимизатор, используемый для обучения</param>
        /// <param name="epochCount">Количество эпох обучения</param>        
        /// <param name="device">Устройство для обучения</param>
        /// <param name="shuffleSampleInMinibatchesPerEpoch">Указывает, что необходимо каждую эпоху перемешивать обучающие примеры для формирования новых минипакетов.</param>
        /// <param name="ruleUpdateLearningRate">Правило обновления скорости обучения. Входные параметры: эпоха, текущая скорость обучения. Выходные: новая скорость обучения.</param>
        /// <param name="actionPerEpoch">Произвольное действие, которое требуется выполнять каждую эпоху. Позволяет прервать процесс тренировки. Входные параметры: эпоха, loss-ошибка, evaluation-ошибка. 
        /// Выходные: true - прервать процесс тренировки, false - продолжить тренировку.
        /// Используется для осуществления логирования, отображения процесса обучения, сохранения промежуточных чекпоинтов модели и т.п.</param>
        /// <returns></returns>
        public static FitResult Fit<T>(this Sequential<T> source,
            IList<IList<T[]>> features,
            IList<T[]> labels,
            int minibatchSize,
            Loss lossFunction,
            Loss evaluationFunction,
            Optimizer optimizer,
            int epochCount,
            DeviceDescriptor device,
            bool shuffleSampleInMinibatchesPerEpoch,
            Func<int, double, double> ruleUpdateLearningRate = null,
            Func<int, double, double, bool> actionPerEpoch = null) where T : IConvertible
        {
            return source.Model.Fit(features, labels, minibatchSize, lossFunction, evaluationFunction, optimizer, epochCount, device, shuffleSampleInMinibatchesPerEpoch, ruleUpdateLearningRate, actionPerEpoch);
        }
        /// <summary>
        /// Обучает модель с двумерным входом. Не применим для обучения реккуретных сетей.
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="trainData">Набор данных для обучения.</param>
        /// <param name="minibatchSize">Размер минипакета</param>
        /// <param name="lossFunction">Функция потерь</param>
        /// <param name="evaluationFunction">Оценочная функция</param>
        /// <param name="optimizer">Оптимизатор, используемый для обучения</param>
        /// <param name="epochCount">Количество эпох обучения</param>        
        /// <param name="device">Устройство для обучения</param>  
        /// <param name="shuffleSampleInMinibatchesPerEpoch">Указывает, что необходимо каждую эпоху перемешивать обучающие примеры для формирования новых минипакетов.</param>
        /// <param name="ruleUpdateLearningRate">Правило обновления скорости обучения. Входные параметры: эпоха, текущая скорость обучения. Выходные: новая скорость обучения.</param>
        /// <param name="actionPerEpoch">Произвольное действие, которое требуется выполнять каждую эпоху. Позволяет прервать процесс тренировки. Входные параметры: эпоха, loss-ошибка, evaluation-ошибка. 
        /// Выходные: true - прервать процесс тренировки, false - продолжить тренировку.
        /// Используется для осуществления логирования, отображения процесса обучения, сохранения промежуточных чекпоинтов модели и т.п.</param>
        /// <returns></returns>
        public static FitResult Fit<T>(this Sequential<T> source,
            IList<Sample2D<T>> trainData,
            int minibatchSize,
            Loss lossFunction,
            Loss evaluationFunction,
            Optimizer optimizer,
            int epochCount,
            DeviceDescriptor device,
            bool shuffleSampleInMinibatchesPerEpoch,
            Func<int, double, double> ruleUpdateLearningRate = null,
            Func<int, double, double, bool> actionPerEpoch = null) where T : IConvertible
        {
            return source.Model.Fit(trainData, minibatchSize, lossFunction, evaluationFunction, optimizer, epochCount, device, shuffleSampleInMinibatchesPerEpoch, ruleUpdateLearningRate, actionPerEpoch);
        }
        /// <summary>
        /// Обучает модель. Не применим для обучения реккуретных сетей.
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="trainData">Набор данных для обучения. Каждый пример должен содержать в начале массива признаки размерностью inputDim, а в конце метки классов размерностью outputDim.
        /// Например inputDim = 3, outputDim = 2: [f1, f2, f3, l1, l2]</param>
        /// <param name="inputDim">Размерность признаков (разрядность)</param>
        /// <param name="minibatchSize">Размер минипакета</param>
        /// <param name="lossFunction">Функция потерь</param>
        /// <param name="evaluationFunction">Оценочная функция</param>
        /// <param name="optimizer">Оптимизатор, используемый для обучения</param>
        /// <param name="epochCount">Количество эпох обучения</param>        
        /// <param name="device">Устройство для обучения</param>       
        /// <param name="ruleUpdateLearningRate">Правило обновления скорости обучения. Входные параметры: эпоха, текущая скорость обучения. Выходные: новая скорость обучения.</param>
        /// <param name="actionPerEpoch">Произвольное действие, которое требуется выполнять каждую эпоху. Позволяет прервать процесс тренировки. Входные параметры: эпоха, loss-ошибка, evaluation-ошибка. 
        /// Выходные: true - прервать процесс тренировки, false - продолжить тренировку.
        /// Используется для осуществления логирования, отображения процесса обучения, сохранения промежуточных чекпоинтов модели и т.п.</param>
        /// <returns></returns>
        public static FitResult Fit<T>(this Sequential<T> source,
            IEnumerable<T[]> trainData,
            int inputDim,
            int minibatchSize,
            Loss lossFunction,
            Loss evaluationFunction,
            Optimizer optimizer,
            int epochCount,
            DeviceDescriptor device,
            Func<int, double, double> ruleUpdateLearningRate = null,
            Func<int, double, double, bool> actionPerEpoch = null) where T : IConvertible
        {
            return source.Model.Fit(trainData, inputDim, minibatchSize, lossFunction, evaluationFunction, optimizer, epochCount, device, ruleUpdateLearningRate, actionPerEpoch);
        }
        /// <summary>
        /// Обучает реккурентную модель.
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="features">Набор последовательностей (признаков). Каждая последовательность может быть переменной длинны, но одинаковой размерности (массивы из которых состоит последовательность, должны иметь одинаковую длину)</param>
        /// <param name="labels">Набор меток. Размерность меток должна быть одинаковая.</param>
        /// <param name="minibatchSize">Размер минипакета</param>
        /// <param name="lossFunction">Функция потерь</param>
        /// <param name="evaluationFunction">Оценочная функция</param>
        /// <param name="optimizer">Оптимизатор, используемый для обучения</param>
        /// <param name="epochCount">Количество эпох обучения</param>        
        /// <param name="device">Устройство для обучения</param>        
        /// <param name="ruleUpdateLearningRate">Правило обновления скорости обучения. Входные параметры: эпоха, текущая скорость обучения. Выходные: новая скорость обучения.</param>
        /// <param name="actionPerEpoch">Произвольное действие, которое требуется выполнять каждую эпоху. Позволяет прервать процесс тренировки. Входные параметры: эпоха, loss-ошибка, evaluation-ошибка. 
        /// Выходные: true - прервать процесс тренировки, false - продолжить тренировку.
        /// Используется для осуществления логирования, отображения процесса обучения, сохранения промежуточных чекпоинтов модели и т.п.</param>
        /// <returns></returns>
        public static FitResult Fit<T>(this Sequential<T> source,
            IEnumerable<IList<T[]>> features,
            IEnumerable<T[]> labels,
            int minibatchSize,
            Loss lossFunction,
            Loss evaluationFunction,
            Optimizer optimizer,
            int epochCount,
            DeviceDescriptor device,
            Func<int, double, double> ruleUpdateLearningRate = null,
            Func<int, double, double, bool> actionPerEpoch = null) where T : IConvertible
        {
            return source.Model.Fit(features, labels, minibatchSize, lossFunction, evaluationFunction, optimizer, epochCount, device, ruleUpdateLearningRate, actionPerEpoch);
        }
        /// <summary>
        /// Обучает модель с двумерным входом. Не применим для обучения реккуретных сетей.
        /// </summary>
        /// <typeparam name="T">Тип данных. Поддерживается <seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="source"></param>
        /// <param name="trainData">Набор данных для обучения.</param>
        /// <param name="minibatchSize">Размер минипакета</param>
        /// <param name="lossFunction">Функция потерь</param>
        /// <param name="evaluationFunction">Оценочная функция</param>
        /// <param name="optimizer">Оптимизатор, используемый для обучения</param>
        /// <param name="epochCount">Количество эпох обучения</param>        
        /// <param name="device">Устройство для обучения</param>          
        /// <param name="ruleUpdateLearningRate">Правило обновления скорости обучения. Входные параметры: эпоха, текущая скорость обучения. Выходные: новая скорость обучения.</param>
        /// <param name="actionPerEpoch">Произвольное действие, которое требуется выполнять каждую эпоху. Позволяет прервать процесс тренировки. Входные параметры: эпоха, loss-ошибка, evaluation-ошибка. 
        /// Выходные: true - прервать процесс тренировки, false - продолжить тренировку.
        /// Используется для осуществления логирования, отображения процесса обучения, сохранения промежуточных чекпоинтов модели и т.п.</param>
        /// <returns></returns>
        public static FitResult Fit<T>(this Sequential<T> source,
            IEnumerable<Sample2D<T>> trainData,
            int minibatchSize,
            Loss lossFunction,
            Loss evaluationFunction,
            Optimizer optimizer,
            int epochCount,
            DeviceDescriptor device,
            Func<int, double, double> ruleUpdateLearningRate = null,
            Func<int, double, double, bool> actionPerEpoch = null) where T : IConvertible
        {
            return source.Model.Fit(trainData, minibatchSize, lossFunction, evaluationFunction, optimizer, epochCount, device, ruleUpdateLearningRate, actionPerEpoch);
        }
        #endregion

    }
}

