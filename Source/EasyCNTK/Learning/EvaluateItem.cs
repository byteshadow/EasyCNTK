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
    /// Represents real and model calculated output values
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public struct EvaluateItem<T> where T:IConvertible
    {
        /// <summary>
        /// Actual Expected Outputs
        /// </summary>
        public IList<T> ExpectedValue { get; set; }
        /// <summary>
        /// Model-calculated output values
        /// </summary>
        public IList<T> EvaluatedValue { get; set; }
        public EvaluateItem(IList<T> expectedValue, IList<T> evaluatedValue)
        {
            if (expectedValue.Count != evaluatedValue.Count)
            {
                throw new ArgumentException($"Несоответсвие размерности ожидаемых({expectedValue.Count}) и оцененных({evaluatedValue.Count}) значений.");
            }
            ExpectedValue = expectedValue;
            EvaluatedValue = evaluatedValue;
        }
    }
}
