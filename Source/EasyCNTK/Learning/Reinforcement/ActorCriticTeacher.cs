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
using System.Globalization;
using System.Linq;
using System.Text;

namespace EasyCNTK.Learning.Reinforcement
{
    /// <summary>
    /// Implements Actor Critic learning mechanism
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class ActorCriticTeacher<T> : AgentTeacher<T> where T : IConvertible
    {
        public ActorCriticTeacher(Environment environment, DeviceDescriptor device) : base(environment, device) { }

        /// <summary>
        /// Teaches an agent whose model is represented by a direct distribution network (non-recurrent) with two outputs (not to be confused with the output dimension). Used when the model operates only with the current state of the environment, not taking into account previous states.
        /// </summary>
        /// <param name="agent">An agent for training, a network of a given architecture with two outputs: 1 output - agent actions, 2 output - average reward when an action is performed (one number)</param>
        /// <param name="iterationCount">Number of learning iterations (eras)</param>
        /// <param name="rolloutCount">The number of runs (in the case of a game - passing the level until the end of the game <seealso cref="Environment.IsTerminated"/> ), which will be completed before the weights are updated.
        /// It can be interpreted as the amount of training data for one era.</param>
        /// <param name="minibatchSize">Minibatch size for training</param>
        /// <param name="actionPerIteration">The arbitrary action that each epoch requires. Allows you to interrupt the training process. Input parameters: era, loss error, evaluation error. 
        /// Weekend: true - interrupt the training process, false - continue the training.
        /// Used for logging, displaying the learning process, saving intermediate model checkpoints, etc.</param>
        /// <param name="gamma">Reward attenuation coefficient (reward) when calculating Discounted reward</param>
        /// <param name="epsilon">The value by which two real numbers must differ in order to be considered different. It is necessary to calculate similar environmental conditions.</param>
        /// <returns></returns>
        public SequentialMultiOutput<T> Teach(SequentialMultiOutput<T> agent, int iterationCount, int rolloutCount, int minibatchSize, Func<int, double[], double[], bool> actionPerIteration = null, double gamma = 0.99, double epsilon = 0.01)
        {
            if (agent.Model.Outputs.Count != 2)
                throw new NotSupportedException("Number of outputs(branches) agent must be equal 2. Other configurations are not supported..");
            if (agent.Model.Outputs[1].Shape.Rank != 1 || agent.Model.Outputs[1].Shape.Dimensions[0] != 1)
                throw new NotSupportedException("The dimension of the second agent output should be equal to 1(output should return a single number). Other configurations are not supported..");

            for (int iteration = 0; iteration < iterationCount; iteration++)
            {
                var data = new LinkedList<(int rollout, int actionNumber, T[] state, T[] action, T reward, T agentReward)>();
                for (int rolloutNumber = 0; rolloutNumber < rolloutCount; rolloutNumber++)
                {
                    int actionNumber = 0;
                    while (!Environment.IsTerminated)
                    {
                        var currentState = Environment.GetCurrentState<T>();
                        var agentOutput = agent.Predict(currentState, Device);
                        var action = agentOutput[0];
                        var agentReward = agentOutput[1][0];
                        var reward = Environment.PerformAction(action);
                        data.AddLast((rolloutNumber, ++actionNumber, currentState, action, reward, agentReward));
                    }
                    Environment.Reset();
                }
                //1 - first, calculate the average reward for each state = baseline, assign the state its average reward (which gives the Environment - baseline) - these will be marks for training the second head
                var baselines = data
                    .GroupBy(p => p.state, new TVectorComparer<T>(epsilon))
                    .ToDictionary(p => p.Key, q => q.Average(z => z.reward.ToDouble(CultureInfo.InvariantCulture)), new TVectorComparer<T>(epsilon));
                var baselineRewards = data
                    .Select(p => baselines[p.state])
                    .ToArray();

                //2 - then count already advatageReward according to the formula: Yi * (reward - agentReward) - these will be marks for training the first head
                var elementTypeCode = data.First.Value.reward.GetTypeCode();
                var advantageReward = new T[data.Count];
                foreach (var rollout in data.GroupBy(p => p.rollout))
                {
                    var steps = rollout.ToList();
                    steps.Sort((a, b) => a.actionNumber > b.actionNumber ? 1 : a.actionNumber < b.actionNumber ? -1 : 0); //ascending actionNumber
                    for (int i = 0; i < steps.Count; i++)
                    {
                        var remainingRewards = steps.GetRange(i, steps.Count - i)
                            .Select(p => Environment.HasRewardOnlyForRollout
                                ? steps[steps.Count - 1].reward.ToDouble(CultureInfo.InvariantCulture) - steps[i].agentReward.ToDouble(CultureInfo.InvariantCulture)
                                : p.reward.ToDouble(CultureInfo.InvariantCulture) - p.agentReward.ToDouble(CultureInfo.InvariantCulture))
                            .Select(p => (T)Convert.ChangeType(p, elementTypeCode))
                            .ToArray();

                        advantageReward[i] = CalculateDiscountedReward(remainingRewards, gamma);
                    }
                }

                var features = data.Select(p => p.state).ToArray();
                var actionLabels = data.Zip(advantageReward, (d, reward) => Multiply(d.action, reward));
                var labels = actionLabels.Zip(baselineRewards, (action, baseline) => new[] { action, new T[] { (T)Convert.ChangeType(baseline, elementTypeCode) } }).ToArray();

                var fitResult = agent.Fit(features,
                                        labels,
                                        minibatchSize,
                                        GetLoss(),
                                        GetEvalLoss(),
                                        GetOptimizer(),
                                        1,
                                        false,
                                        Device);
                data.Clear();
                var needStop = actionPerIteration?.Invoke(iteration, fitResult.Select(p => p.LossError).ToArray(), fitResult.Select(p => p.EvaluationError).ToArray());
                if (needStop.HasValue && needStop.Value)
                    break;
            }
            return agent;
        }
        /// <summary>
        /// Teaches an agent whose model is represented by a recurrent network with two outputs (not to be confused with the output dimension). It is used when the model operates with a chain of environmental states.
        /// </summary>
        /// <param name="agent">An agent for training, a network of a given architecture with two outputs: 1 output - agent actions, 2 output - average reward when an action is performed (one number)</param>
        /// <param name="iterationCount">Number of learning iterations (eras)</param>
        /// <param name="rolloutCount">The number of runs (in the case of a game - passing the level until the end of the game <seealso cref="Environment.IsTerminated"/> ), which will be completed before the weights are updated.
        /// It can be interpreted as the amount of training data for one era.</param>
        /// <param name="minibatchSize">Minibatch size for training</param>
        /// <param name="sequenceLength">Sequence length: a chain of previous state environments on each action.</param>
        /// <param name="actionPerIteration">The arbitrary action that each epoch requires. Allows you to interrupt the training process. Input parameters: era, loss error, evaluation error. 
        /// Weekend: true - interrupt the training process, false - continue the training.
        /// Used for logging, displaying the learning process, saving intermediate model checkpoints, etc.</param>
        /// <param name="gamma">Reward attenuation coefficient (reward) when calculating Discounted reward</param>
        /// <param name="epsilon">The value by which two real numbers must differ in order to be considered different. It is necessary to calculate similar environmental conditions.</param>
        /// <returns></returns>
        public SequentialMultiOutput<T> Teach(SequentialMultiOutput<T> agent, int iterationCount, int rolloutCount, int minibatchSize, int sequenceLength, Func<int, double[], double[], bool> actionPerIteration = null, double gamma = 0.99, double epsilon = 0.01)
        {
            if (agent.Model.Outputs.Count != 2)
                throw new NotSupportedException("Number of outputs(branches) agent must be equal 2. Other configurations are not supported..");
            if (agent.Model.Outputs[1].Shape.Rank != 1 || agent.Model.Outputs[1].Shape.Dimensions[0] != 1)
                throw new NotSupportedException("The dimension of the second agent output should be equal to 1(output should return a single number). Other configurations are not supported..");

            for (int iteration = 0; iteration < iterationCount; iteration++)
            {
                var data = new List<(int rollout, int actionNumber, T[] state, T[] action, T reward, T agentReward)>();
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
                        var agentOutput = agent.Predict(sequenceStates, Device);
                        var action = agentOutput[0];
                        var agentReward = agentOutput[1][0];
                        var reward = Environment.PerformAction(action);
                        data.Add((rolloutNumber, ++actionNumber, currentState, action, reward, agentReward));
                    }
                    Environment.Reset();
                }
                //1 - first, calculate the average reward for each state = baseline, assign the state its average reward (which gives the Environment - baseline) - these will be marks for training the second head
                var baselines = data
                    .GroupBy(p => p.state, new TVectorComparer<T>(epsilon))
                    .ToDictionary(p => p.Key, q => q.Average(z => z.reward.ToDouble(CultureInfo.InvariantCulture)), new TVectorComparer<T>(epsilon));
                var baselineRewards = data
                    .Select(p => baselines[p.state])
                    .ToArray();

                //2 - then count already advatageReward according to the formula: Yi * (reward - agentReward) - these will be marks for training the first head
                var elementTypeCode = data[0].reward.GetTypeCode();
                var advantageReward = new T[data.Count];
                foreach (var rollout in data.GroupBy(p => p.rollout))
                {
                    var steps = rollout.ToList();
                    steps.Sort((a, b) => a.actionNumber > b.actionNumber ? 1 : a.actionNumber < b.actionNumber ? -1 : 0); //ascending actionNumber
                    for (int i = 0; i < steps.Count; i++)
                    {
                        var remainingRewards = steps.GetRange(i, steps.Count - i)
                            .Select(p => Environment.HasRewardOnlyForRollout
                                ? steps[steps.Count - 1].reward.ToDouble(CultureInfo.InvariantCulture) - steps[i].agentReward.ToDouble(CultureInfo.InvariantCulture)
                                : p.reward.ToDouble(CultureInfo.InvariantCulture) - p.agentReward.ToDouble(CultureInfo.InvariantCulture))
                            .Select(p => (T)Convert.ChangeType(p, elementTypeCode))
                            .ToArray();

                        advantageReward[i] = CalculateDiscountedReward(remainingRewards, gamma);
                    }
                }

                var features = new List<IList<T[]>>();
                var labels = new List<T[][]>();
                var dataWithAdvantageRewardAndBaselines = data
                    .Zip(advantageReward, (dat, reward) => (dat, reward))
                    .Zip(baselineRewards, (first, baseline) => (first.dat, first.reward, baseline))
                    .GroupBy(p => p.dat.rollout);
                foreach (var rollout in dataWithAdvantageRewardAndBaselines)
                {
                    var steps = rollout.ToList();
                    steps.Sort((a, b) => a.dat.actionNumber > b.dat.actionNumber ? 1 : a.dat.actionNumber < b.dat.actionNumber ? -1 : 0); //ascending actionNumber
                    for (int i = 0; i < steps.Count; i++)
                    {
                        if (i < sequenceLength)
                        {
                            features.Add(steps
                                .GetRange(0, i + 1)
                                .Select(p => p.dat.state)
                                .ToArray());
                        }
                        else
                        {
                            features.Add(steps
                                .GetRange(i - sequenceLength, sequenceLength)
                                .Select(p => p.dat.state)
                                .ToArray());
                        }
                        labels.Add(new[]
                        {
                            Multiply(steps[i].dat.action, steps[i].reward),
                            new T[] { (T)Convert.ChangeType(steps[i].baseline, elementTypeCode) }
                        });
                    }
                }

                var fitResult = agent.Fit(features,
                                        labels,
                                        minibatchSize,
                                        GetLoss(),
                                        GetEvalLoss(),
                                        GetOptimizer(),
                                        1,
                                        false,
                                        Device);
                data.Clear();
                var needStop = actionPerIteration?.Invoke(iteration, fitResult.Select(p => p.LossError).ToArray(), fitResult.Select(p => p.EvaluationError).ToArray());
                if (needStop.HasValue && needStop.Value)
                    break;
            }
            return agent;
        }

    }
}
