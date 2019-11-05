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
    /// FSAdaGrad Optimizer. Analogue<seealso cref="Adam"/>
    /// </summary>
    public sealed class FSAdaGrad : Optimizer
    {
        private double _l1RegularizationWeight;
        private double _l2RegularizationWeight;
        private double _gradientClippingThresholdPerSample;
        private double _momentum;
        private double _varianceMomentumSchedule;
        private bool _unitGain;
        public override double LearningRate { get; }
        public override int MinibatchSize { get; set; }

        /// <summary>
        /// Initializes FSAdaGrad Optimizer
        /// </summary>
        /// <param name="learningRate">Learning speed</param>
        /// <param name="momentum">Moment</param>
        /// <param name="minibatchSize">The mini-packet size is required by CNTK to scale optimizer parameters for more effective training. If equal to 0, then the mitibatch size will be used during training.</param>
        /// <param name="l1RegularizationWeight">Coefficient L1 of norm, if 0 - regularization is not applied</param>
        /// <param name="l2RegularizationWeight">Coefficient L2 of norm, if 0 - regularization is not applied</param>
        /// <param name="gradientClippingThresholdPerSample">The gradient cutoff threshold for each training example is used primarily to combat the explosive gradient in deep recursive networks.
        /// The default is set to<seealso cref="double.PositiveInfinity"/> - отсечение не используется. Для использования установите необходимый порог.</param>       
        /// <param name="varianceMomentumSchedule">&quot;Beta2&quot; parameter in the formula for calculating the moment: http://ruder.io/optimizing-gradient-descent/index.html#adam (similar to FSAdaGrad)</param>
        /// <param name="unitGain">Indicates that the torque is used in gain mode.</param>   
        public FSAdaGrad(double learningRate,
            double momentum,
            int minibatchSize = 0,
            double l1RegularizationWeight = 0,
            double l2RegularizationWeight = 0,
            double gradientClippingThresholdPerSample = double.PositiveInfinity,            
            double varianceMomentumSchedule = 0.9999986111120757,
            bool unitGain = true)
        {
            LearningRate = learningRate;
            _momentum = momentum;
            _l1RegularizationWeight = l1RegularizationWeight;
            _l2RegularizationWeight = l2RegularizationWeight;
            _gradientClippingThresholdPerSample = gradientClippingThresholdPerSample;            
            _varianceMomentumSchedule = varianceMomentumSchedule;
            _unitGain = unitGain;
            MinibatchSize = minibatchSize;
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
            return CNTKLib.FSAdaGradLearner(new ParameterVector((ICollection)learningParameters),
                new TrainingParameterScheduleDouble(LearningRate, (uint)MinibatchSize),
                new TrainingParameterScheduleDouble(_momentum, (uint)MinibatchSize),
                _unitGain,
                new TrainingParameterScheduleDouble(_varianceMomentumSchedule, (uint)MinibatchSize),
                learningOptions);
        }
    }
}
