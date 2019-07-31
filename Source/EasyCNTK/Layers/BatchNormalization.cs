//
// Copyright (c) Stanislav Grigoriev. All rights reserved.
// grigorievstas9@gmail.com 
// https://github.com/StanislavGrigoriev/EasyCNTK
//
// Copyright (c) Microsoft. All rights reserved.
//
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//

using System.Runtime.Serialization;
using CNTK;

namespace EasyCNTK.Layers
{
    /// <summary>
    /// Implements the batch normalization layer
    /// </summary>
    public sealed class BatchNormalization : Layer
    {
        public BatchNormalization() { }

        public BatchNormalization(SerializationInfo info, StreamingContext context) { }

        private static Function CreateBatchNorm(Function input, DeviceDescriptor device)
        {
            var scale = new Parameter(input.Output.Shape, input.Output.DataType, 1, device);
            var bias = new Parameter(input.Output.Shape, input.Output.DataType, 0, device);
            var runningMean = new Constant(input.Output.Shape, input.Output.DataType, 0, device);
            var runningInvStd = new Constant(input.Output.Shape, input.Output.DataType, 0, device);
            var runningCount = new Constant(new[] { 1 }, input.Output.DataType, 0, device);
            return CNTKLib.BatchNormalization(input.Output, scale, bias, runningMean, runningInvStd, runningCount, false);
        }

        /// <summary>
        /// Creates a layer of batch normalization
        /// </summary>
        /// <param name = "input"> </param>
        /// <param name = "device"> </param>
        /// <returns> </returns>
        public static Function Build(Function input, DeviceDescriptor device)
        {
            return CreateBatchNorm(input, device);
        }
        public override Function Create(Function input, DeviceDescriptor device)
        {
            return CreateBatchNorm(input, device);
        }

        public override string GetDescription()
        {
            return "BN";
        }

        public override void GetObjectData(SerializationInfo info, StreamingContext context)
        {

        }
    }
}
