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
    /// Compares two double numbers with a given error
    /// </summary>
    public class DoubleComparer : IComparer<double>
    {
        /// <summary>
        /// Error. Sets the minimum by which two numbers must differ in order to be considered different.
        /// </summary>
        public double Epsilon { get; }
        /// <summary>
        /// Creates a comparator instance
        /// </summary>
        /// <param name="epsilon">Error. Sets the minimum by which two numbers must differ in order to be considered different.</param>
        public DoubleComparer(double epsilon = 0.01)
        {
            Epsilon = epsilon;
        }

        public int Compare(double x, double y)
        {
            if (Math.Abs(x - y) < Epsilon)
            {
                return 0;
            }
            if (x - y > 0)
            {
                return 1;
            }
            return -1;
        }
    }
}
