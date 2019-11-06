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
    /// Optimizer RMSProp. Analogue <seealso cref="AdaDelta"/> . Source: http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    /// </summary>
    public sealed class RMSProp : Optimizer
    {
        private double _l1RegularizationWeight;
        private double _l2RegularizationWeight;
        private double _gradientClippingThresholdPerSample;
        private double _gamma;
        private double _inc;
        private double _dec;
        private double _max;
        private double _min;
        public override double LearningRate { get; }
        public override int MinibatchSize { get; set; }

        /// <summary>
        /// Initializes the RMSProp Optimizer 
        /// </summary>
        /// <param name="learningRate">Learning speed</param>
        /// <param name="minibatchSize">The mini-packet size is required by CNTK to scale optimizer parameters for more effective training. If equal to 0, then the mitibatch size will be used during training.</param>
        /// <param name="gamma">Gain for the previous gradient. Must be within [0; 1]</param>
        /// <param name="increment">The rate of increase in learning speed. Must be greater than 1. Default 5% increase</param>
        /// <param name="decrement">The rate of decrease in learning speed. Must be within [0; 1]. 5% reduction by default</param>
        /// <param name="max">Maximum learning speed. Must be greater than 0 and min</param>
        /// <param name="min">Minimum learning speed. Must be greater than 0 and less than max</param>
        /// <param name="l1RegularizationWeight">Coefficient L1 of norm, if 0 - regularization is not applied</param>
        /// <param name="l2RegularizationWeight">Coefficient L2 of norm, if 0 - regularization is not applied</param>
        /// <param name="gradientClippingThresholdPerSample">The gradient cutoff threshold for each training example is used primarily to combat the explosive gradient in deep recursive networks.
        /// The default is set to<seealso cref="double.PositiveInfinity"/> - clipping is not used. To use, set the required threshold..</param>
        public RMSProp(double learningRate,
            int minibatchSize = 0,
            double gamma = 0.95,
            double increment = 1.05,
            double decrement = 0.95,
            double max = 0.2,
            double min = 1e-08,            
            double l1RegularizationWeight = 0,
            double l2RegularizationWeight = 0,
            double gradientClippingThresholdPerSample = double.PositiveInfinity)
        {
            LearningRate = learningRate;
            _gamma = gamma;
            _inc = increment;
            _dec = decrement;
            _max = max;
            _min = min;
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
            return CNTKLib.RMSPropLearner(new ParameterVector((ICollection)learningParameters),
                new TrainingParameterScheduleDouble(LearningRate, (uint)MinibatchSize),
                _gamma,
                _inc,
                _dec,
                _max,
                _min,
                true,
                learningOptions);
        }
    }
}
