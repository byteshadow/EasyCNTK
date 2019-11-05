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
using System;

namespace EasyCNTK.LossFunctions
{
    /// <summary>
    /// Error function for binary, single-class and multi-class classification. It can be used with a Softmax output, and a classification condition is created: the class must have a probability above a given threshold, otherwise it will not be classified.
    /// </summary>
    public sealed class ClassificationError : Loss
    {
        private double _thresholdValue;
        /// <summary>
        /// Error function for binary, single-class and multi-class classification. It can be used with a Softmax output, and a classification condition is created: the class must have a probability above a given threshold, otherwise it will not be classified.
        /// </summary>
        /// <param name="threshold">The threshold value for the actual value of the output of the neural network, below which the class is not recognized. In other words, this is the minimum probability that the classifier must give for a particular class so that this class is considered as recognized.</param>
        public ClassificationError(double threshold = 0.5)
        {
            if (threshold <= 0 || threshold >= 1) throw new ArgumentOutOfRangeException("threshold", "Порог должен быть в диапазоне: (0;1)");
            _thresholdValue = threshold;
        }
        public override Function GetLoss(Variable prediction, Variable targets, DeviceDescriptor device)
        {
            var threshold = new Constant(prediction.Shape, prediction.DataType, _thresholdValue, device);

            var predictionLabel = CNTKLib.Less(threshold, prediction);
            var loss = CNTKLib.NotEqual(predictionLabel, targets);

            return loss;
        }
    }
}
