//
// Copyright (c) Stanislav Grigoriev. All rights reserved.
// grigorievstas9@gmail.com 
// https://github.com/StanislavGrigoriev/EasyCNTK
//
// Copyright (c) Microsoft. All rights reserved.
//
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//
using System.Collections;
using System.Collections.Generic;
using CNTK;

namespace EasyCNTK.Learning.Optimizers
{
    /// <summary>
    /// Optizer. Implements stochastic gradient descent (SGD)
    /// </summary>
    public sealed class SGD : Optimizer
    {
        private double _l1RegularizationWeight;
        private double _l2RegularizationWeight;
        private double _gradientClippingThresholdPerSample;

        public override double LearningRate { get; }
        public override int MinibatchSize { get => throw new System.NotImplementedException(); set => throw new System.NotImplementedException(); }

        /// <summary>
        /// Initializes the Stochastic Gradient Descent (SGD) optimizer
        /// </summary>
        /// <param name="learningRate">Learning speed</param>
        /// <param name="minibatchSize">The mini-packet size is required by CNTK to scale optimizer parameters for more effective training. If equal to 0, then the mitibatch size will be used during training.</param>
        /// <param name="l1RegularizationWeight">Coefficient L1 of norm, if 0 - regularization is not applied</param>
        /// <param name="l2RegularizationWeight">Coefficient L2 of norm, if 0 - regularization is not applied</param>
        /// <param name="gradientClippingThresholdPerSample">The gradient cutoff threshold for each training example is used primarily to combat the explosive gradient in deep recursive networks.
        /// The default is set to<seealso cref="double.PositiveInfinity"/> - clipping is not used. To use, set the required threshold..</param>
        public SGD(double learningRate,
            int minibatchSize = 0,
            double l1RegularizationWeight = 0,
            double l2RegularizationWeight = 0,
            double gradientClippingThresholdPerSample = double.PositiveInfinity)
        {
            LearningRate = learningRate;
            MinibatchSize = minibatchSize;
            _l1RegularizationWeight = l1RegularizationWeight;
            _l2RegularizationWeight = l2RegularizationWeight;
            _gradientClippingThresholdPerSample = gradientClippingThresholdPerSample;
        }
        public override Learner GetOptimizer(IList<Parameter> learningParameters)
        {
            var learningOptions = new AdditionalLearningOptions()
            {
                l1RegularizationWeight = _l1RegularizationWeight,
                l2RegularizationWeight = _l2RegularizationWeight,
                gradientClippingWithTruncation = _gradientClippingThresholdPerSample != double.PositiveInfinity,
                gradientClippingThresholdPerSample = _gradientClippingThresholdPerSample
            };
            return CNTKLib.SGDLearner(new ParameterVector((ICollection)learningParameters),
                new TrainingParameterScheduleDouble(LearningRate, (uint)MinibatchSize),                
                learningOptions);
        }
    }
}
