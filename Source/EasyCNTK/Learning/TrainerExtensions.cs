using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using CNTK;

namespace EasyCNTK.Learning
{
    public static class TrainerExtensions
    {
        public static FitResult Fit(this Trainer trainer,
            DeviceDescriptor device,
            MinibatchSource miniBatchSource, uint miniBatchSize,
            Dictionary<Variable, StreamInformation> streamInfos,
            Learner learner,
            double learningRate,
            int epochs,
            Func<int, double, double> ruleUpdateLearningRate = null,
            Func<int, uint, uint> ruleUpdateBatchSize = null,
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
                miniBatchSize = ruleUpdateBatchSize?.Invoke(epochCount, miniBatchSize) ?? miniBatchSize;

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

    }
}
