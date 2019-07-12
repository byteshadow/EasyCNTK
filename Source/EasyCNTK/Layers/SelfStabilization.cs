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

namespace EasyCNTK.Layers
{
    /// <summary>
    /// Implements a self-stabilization layer for choosing the optimal learning speed. A source: https://github.com/Microsoft/CNTK/blob/release/latest/Examples/TrainingCSharp/Common/LSTMSequenceClassifier.cs
    /// </summary>
    public sealed class SelfStabilization : Layer
    {
        private readonly string _name;
        private readonly DataType _dataType;

        private static Function SelfStabilize(Function input, DeviceDescriptor device, string name, DataType dataType)
        {
            var isFloatType = dataType == DataType.Float || input.Output.DataType == DataType.Float;

            Constant f, fInv;
            if (isFloatType)
            {
                f = Constant.Scalar(4.0f, device);
                fInv = Constant.Scalar(f.DataType, 1.0 / 4.0f);
            }
            else
            {
                f = Constant.Scalar(4.0, device);
                fInv = Constant.Scalar(f.DataType, 1.0 / 4.0);
            }

            var beta = CNTKLib.ElementTimes(
                fInv,
                CNTKLib.Log(
                    Constant.Scalar(f.DataType, 1.0) +
                    CNTKLib.Exp(CNTKLib.ElementTimes(f,
                        new Parameter(new NDShape(), f.DataType, 0.99537863 /* 1/f*ln(e^f-1)*/, device, "alpha")))),
                "beta");
            return Function.Alias(CNTKLib.ElementTimes(beta, input), name);
        }

        /// <summary>
        /// Creates a layer of self-stabilization for choosing the optimal learning speed.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="device"></param>
        /// <param name="name"></param>
        /// <param name="dataType"></param>
        /// <returns></returns>
        public static Function Build(Function input, DeviceDescriptor device, string name,DataType dataType = DataType.Unknown)
        {
            return SelfStabilize(input, device, name, dataType);
        }

        public override Function Create(Function input, DeviceDescriptor device)
        {
            return SelfStabilize(input, device, _name, _dataType);
        }

        /// <summary>
        /// Creates a self-stabilization layer for choosing the optimal learning speed.
        /// </summary>
        /// <param name="dataType"></param>
        /// <param name="name"></param>
        public SelfStabilization(DataType dataType = DataType.Unknown, string name = "SelfStabilizer")
        {
            _dataType = dataType;
            _name = name;
        }

        public override string GetDescription()
        {
            return "SS";
        }
    }
}
