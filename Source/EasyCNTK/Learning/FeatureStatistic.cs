//
// Copyright (c) Stanislav Grigoriev. All rights reserved.
// grigorievstas9@gmail.com 
// https://github.com/StanislavGrigoriev/EasyCNTK
//
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//
using System.Collections.Generic;

namespace EasyCNTK.Learning
{
    /// <summary>
    /// Represents variable statistics in a dataset
    /// </summary>
    public class FeatureStatistic
    {
        /// <summary>
        /// Property name of the class representing the characteristic / variable
        /// </summary>
        public string FeatureName { get; set; }    
        /// <summary>
        /// Average value
        /// </summary>
        public double Average { get; set; }
        /// <summary>
        /// Median
        /// </summary>
        public double Median { get; set; }
        /// <summary>
        /// The minimum value of the variable
        /// </summary>
        public double Min { get; set; }
        /// <summary>
        /// The maximum value of the variable
        /// </summary>
        public double Max { get; set; }
        /// <summary>
        /// Standard deviation
        /// </summary>
        public double StandardDeviation { get; set; }
        /// <summary>
        /// Dispersion
        /// </summary>
        public double Variance { get; set; }
        /// <summary>
        /// Mean Absolute Deviation
        /// </summary>
        public double MeanAbsoluteDeviation { get; set; }
        /// <summary>
        /// An ordered list of unique variable values. Key - the value of the variable, Value - the number of variables with this value
        /// </summary>
        public SortedList<double, int> UniqueValues { get; set; } 
        /// <summary>
        /// Initializes class
        /// </summary>
        /// <param name="epsilon">Error. Sets the minimum by which two numbers must differ in order to be considered different.</param>
        public FeatureStatistic(double epsilon = 0.01)
        {
            UniqueValues = new SortedList<double, int>(new DoubleComparer(epsilon));            
        }

        public override string ToString()
        {
            return $"Name: {FeatureName} Min: {Min:F5} Max: {Max:F5} Average: {Average:F5} MAD: {MeanAbsoluteDeviation:F5} Variance: {Variance:F5} StdDev: {StandardDeviation:F5}";
        }
    }
}