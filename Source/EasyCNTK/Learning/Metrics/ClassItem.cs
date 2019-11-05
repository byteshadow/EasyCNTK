//
// Copyright (c) Stanislav Grigoriev. All rights reserved.
// grigorievstas9@gmail.com 
// https://github.com/StanislavGrigoriev/EasyCNTK
//
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//

namespace EasyCNTK.Learning.Metrics
{
    /// <summary>
    /// Contains additional classification metrics for a specific class
    /// </summary>
    public class ClassItem
    {
        /// <summary>
        /// The index of the position in the output vector of the model assigned to a certain class
        ///</summary>
        public int Index { get; set; }
        /// <summary>
        /// The accuracy with which the model defines this class. It is calculated by the formula: accuracy = [number of correctly defined examples of this class] / [number of examples classified as this class]
        /// </summary>
        public double Precision { get; set; }
        /// <summary>
        /// The completeness with which the model defines this class. It is calculated by the formula: completeness = [number of correctly defined examples of this class] / [number of all examples of this class]
        /// </summary>
        public double Recall { get; set; }
        /// <summary>
        /// Harmonic between <seealso cref="Precision"/> and <seealso cref="Recall"/> . Calculated by the formula: F1Score = 2 * Precision * Recall / (Precision + Recall)
        /// </summary>
        public double F1Score { get; set; }
        /// <summary>
        /// The proportion of examples of this class in the entire dataset
        /// </summary>
        public double Fraction { get; set; }        
    }
}
