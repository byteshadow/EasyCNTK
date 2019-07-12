//
// Copyright (c) Stanislav Grigoriev. All rights reserved.
// grigorievstas9@gmail.com 
// https://github.com/StanislavGrigoriev/EasyCNTK
//
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//
using System;
using System.Collections.Generic;

namespace EasyCNTK.Learning
{
    /// <summary>
    /// Represents the result of a single training session.
    /// </summary>
    public class FitResult
    {
        /// <summary>
        /// The average error of the loss function on the results of the training session
        /// </summary>
        public double LossError { get; set; }

        /// <summary>
        /// The average error of the loss function on the results of the training session
        /// </summary>
        public double EvaluationError { get; set; }

        /// <summary>
        /// Duration of the training session
        /// </summary>
        public TimeSpan Duration { get; set; }

        /// <summary>
        /// Number of learning epochs
        /// </summary>
        public int EpochCount { get; set; }

        /// <summary>
        /// Error curve of the loss function in the learning process
        /// </summary>

        public List<double> LossCurve { get; set; }

        /// <summary>
        /// Error curve of the evaluation function in the learning process
        /// </summary>
        public List<double> EvaluationCurve { get; set; }
    }
}
