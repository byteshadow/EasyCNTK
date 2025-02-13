//
// Copyright (c) Stanislav Grigoriev. All rights reserved.
// grigorievstas9@gmail.com 
// https://github.com/StanislavGrigoriev/EasyCNTK
//
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//
using CNTK;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace EasyCNTK.Learning.Reinforcement
{
    /// <summary>
    /// Implements a learning mechanism using the Policy Gradients method
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class PolicyGradientsTeacher<T> : AgentTeacher<T> where T : IConvertible
    {
        public PolicyGradientsTeacher(Environment environment, DeviceDescriptor device) : base(environment, device) { }

        /// <summary>
        /// Teaches an agent whose model is represented by a direct distribution network (non-recurrent). Used when the model operates only with the current state of the environment, not taking into account previous states.
        /// </summary>
        /// <param name="agent">Agent for training, a network of a given architecture</param>
        /// <param name="iterationCount">Number of learning iterations (eras)</param>
        /// <param name="rolloutCount">The number of runs (in the case of a game - passing the level until the end of the game <seealso cref="Environment.IsTerminated"/> ), which will be completed before the weights are updated.
        /// It can be interpreted as the amount of training data for one era.</param>
        /// <param name="minibatchSize">Minibatch size for training</param>
        /// <param name="actionPerIteration">The arbitrary action that each epoch requires. Allows you to interrupt the training process. Input parameters: era, loss error, evaluation error. 
        /// Weekend: true - interrupt the training process, false - continue the training.
        /// Used for logging, displaying the learning process, saving intermediate model checkpoints, etc.</param>
        /// <param name="gamma">Reward attenuation coefficient (reward) when calculating Discounted reward</param>
        /// <returns></returns>
        public Sequential<T> Teach(Sequential<T> agent, int iterationCount, int rolloutCount, int minibatchSize, Func<int, double, double, bool> actionPerIteration = null, double gamma = 0.99)
        {
            for (int iteration = 0; iteration < iterationCount; iteration++)
            {
                var data = new LinkedList<(int rollout, int actionNumber, T[] state, T[] action, T reward)>();
                for (int rolloutNumber = 0; rolloutNumber < rolloutCount; rolloutNumber++)
                {
                    int actionNumber = 0;
                    while (!Environment.IsTerminated)
                    {
                        var currentState = Environment.GetCurrentState<T>();
                        var action = agent.Predict(currentState, Device);
                        var reward = Environment.PerformAction(action);
                        data.AddLast((rolloutNumber, ++actionNumber, currentState, action, reward));
                    }
                    Environment.Reset();
                }
                var discountedRewards = new T[data.Count];
                foreach (var rollout in data.GroupBy(p => p.rollout))
                {
                    var steps = rollout.ToList();
                    steps.Sort((a, b) => a.actionNumber > b.actionNumber ? 1 : a.actionNumber < b.actionNumber ? -1 : 0); //ascending actionNumber
                    for (int i = 0; i < steps.Count; i++)
                    {
                        var remainingRewards = steps.GetRange(i, steps.Count - i)
                            .Select(p => Environment.HasRewardOnlyForRollout ? steps[steps.Count - 1].reward : p.reward)
                            .ToArray();
                        discountedRewards[i] = CalculateDiscountedReward(remainingRewards, gamma);
                    }
                }

                var features = data.Select(p => p.state);
                var labels = data.Zip(discountedRewards, (d, reward) => Multiply(d.action, reward));
                var dataset = features.Zip(labels, (f, l) => f.Concat(l).ToArray()).ToArray();
                var inputDim = features.FirstOrDefault().Length;

                var fitResult = agent.Fit(dataset,
                                        inputDim,
                                        minibatchSize,
                                        GetLoss()[0],
                                        GetEvalLoss()[0],
                                        GetOptimizer()[0],
                                        1,
                                        false,
                                        Device);
                data.Clear();
                var needStop = actionPerIteration?.Invoke(iteration, fitResult.LossError, fitResult.EvaluationError);
                if (needStop.HasValue && needStop.Value)
                    break;
            }
            return agent;
        }
        /// <summary>
        /// Teaches an agent whose model is represented by a recurrent network. It is used when the model operates with a chain of environmental states.
        /// </summary>
        /// <param name="agent">Agent for training, a network of a given architecture</param>
        /// <param name="iterationCount">Number of learning iterations (eras)</param>
        /// <param name="rolloutCount">The number of runs (in the case of a game - passing the level until the end of the game <seealso cref="Environment.IsTerminated"/> ), which will be completed before the weights are updated.
        /// It can be interpreted as the amount of training data for one era.</param>
        /// <param name="minibatchSize">Minibatch size for training</param>
        /// <param name="sequenceLength">Sequence length: a chain of previous state environments on each action.</param>
        /// <param name="actionPerIteration">The arbitrary action that each epoch requires. Allows you to interrupt the training process. Input parameters: era, loss error, evaluation error. 
        /// Weekend: true - interrupt the training process, false - continue the training.
        /// Used for logging, displaying the learning process, saving intermediate model checkpoints, etc.</param>
        /// <param name="gamma">Reward attenuation coefficient (reward) when calculating Discounted reward</param>
        /// <returns></returns>
        public Sequential<T> Teach(Sequential<T> agent, int iterationCount, int rolloutCount, int minibatchSize, int sequenceLength, Func<int, double, double, bool> actionPerIteration = null, double gamma = 0.99)
        {
            for (int iteration = 0; iteration < iterationCount; iteration++)
            {
                var data = new List<(int rollout, int actionNumber, T[] state, T[] action, T reward)>();
                for (int rolloutNumber = 0; rolloutNumber < rolloutCount; rolloutNumber++)
                {
                    int actionNumber = 0;
                    while (!Environment.IsTerminated)
                    {
                        var currentState = Environment.GetCurrentState<T>();
                        var sequence = actionNumber < sequenceLength
                            ? data.GetRange(data.Count - actionNumber, actionNumber)
                            : data.GetRange(data.Count - sequenceLength - 1, sequenceLength - 1);
                        var sequenceStates = sequence
                            .Select(p => p.state)
                            .ToList();
                        sequenceStates.Add(currentState);
                        var action = agent.Predict(sequenceStates, Device);
                        var reward = Environment.PerformAction(action);
                        data.Add((rolloutNumber, ++actionNumber, currentState, action, reward));
                    }
                    Environment.Reset();
                }
                var discountedRewards = new T[data.Count];
                foreach (var rollout in data.GroupBy(p => p.rollout))
                {
                    var steps = rollout.ToList();
                    for (int i = 0; i < steps.Count; i++)
                    {
                        var remainingRewards = steps.GetRange(i, steps.Count - i)
                            .Select(p => Environment.HasRewardOnlyForRollout ? steps[steps.Count - 1].reward : p.reward)
                            .ToArray();
                        discountedRewards[i] = CalculateDiscountedReward(remainingRewards, gamma);
                    }
                }

                var features = new List<IList<T[]>>();
                var labels = new List<T[]>();
                var dataWithDiscountedReward = data.Zip(discountedRewards, (dat, reward) => (dat, reward)).GroupBy(p => p.dat.rollout);
                foreach (var rollout in dataWithDiscountedReward)
                {
                    var steps = rollout.ToList();
                    steps.Sort((a, b) => a.dat.actionNumber > b.dat.actionNumber ? 1 : a.dat.actionNumber < b.dat.actionNumber ? -1 : 0); //ascending actionNumber
                    for (int i = 0; i < steps.Count; i++)
                    {
                        if (i < sequenceLength)
                        {
                            features.Add(steps.GetRange(0, i + 1).Select(p => p.dat.state).ToArray());
                        }
                        else
                        {
                            features.Add(steps.GetRange(i - sequenceLength, sequenceLength).Select(p => p.dat.state).ToArray());
                        }
                        labels.Add(Multiply(steps[i].dat.action, steps[i].reward));
                    }
                }

                var fitResult = agent.Fit(features,
                                        labels,
                                        minibatchSize,
                                        GetLoss()[0],
                                        GetEvalLoss()[0],
                                        GetOptimizer()[0],
                                        1,
                                        Device);
                data.Clear();
                var needStop = actionPerIteration?.Invoke(iteration, fitResult.LossError, fitResult.EvaluationError);
                if (needStop.HasValue && needStop.Value)
                    break;
            }
            return agent;
        }

    }
}
