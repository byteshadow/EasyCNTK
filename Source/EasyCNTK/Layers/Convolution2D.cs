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
using EasyCNTK.ActivationFunctions;

namespace EasyCNTK.Layers
{
    /// <summary>
    /// Implements a convolutional layer for a two-dimensional vector
    /// </summary>
    public sealed class Convolution2D : Layer
    {
        private int _kernelWidth;
        private int _kernelHeight;
        private int _outFeatureMapCount;
        private int _hStride;
        private int _vStride;
        private Padding _padding;
        private ActivationFunction _activationFunction;
        private string _name;
        /// <summary>
        /// Adds a convolution layer for a two-dimensional vector. If the previous layer has a non-two-dimensional output, an exception is thrown
        /// </summary>
        /// <param name="kernelWidth">Convolution core width (columns in a two-dimensional matrix)</param>
        /// <param name="kernelHeight">Convolution core height (rows in a two-dimensional matrix)</param>
        /// <param name="outFeatureMapCount">Bit depth of the output cell after convolution</param>
        /// <param name="activationFunction">Activation function for the output layer. If not required, pass null</param>
        /// <param name="hStride">The step of moving down the convolution window horizontally (along the matrix columns)</param>
        /// <param name="vStride">Step of shifting the convolution window vertically (along the rows of the matrix)</param>
        /// <param name="padding">Fill when using convolution</param>
        /// <param name="name"></param>
        public static Function Build(Variable input, int kernelWidth, int kernelHeight, DeviceDescriptor device, int outFeatureMapCount = 1, int hStride = 1, int vStride = 1, Padding padding = Padding.Valid, ActivationFunction activationFunction = null, string name = "Conv2D")
        {
            bool[] paddingVector = null;
            if (padding == Padding.Valid)
            {
                paddingVector = new bool[] { false, false, false };
            }
            if (padding == Padding.Same)
            {
                paddingVector = new bool[] { true, true, false };
            }
            
            var convMap = new Parameter(new int[] { kernelWidth, kernelHeight, 1, outFeatureMapCount }, input.DataType, CNTKLib.GlorotUniformInitializer(), device);
            var convolution = CNTKLib.Convolution(convMap, input, new int[] { hStride, vStride, 1 }, new bool[] { true }, paddingVector);
            var activatedConvolution = activationFunction?.ApplyActivationFunction(convolution, device) ?? convolution;

            return Function.Alias(activatedConvolution, name);
        }

        public override Function Create(Function input, DeviceDescriptor device)
        {
            return Build(input, _kernelWidth, _kernelHeight, device, _outFeatureMapCount, _hStride, _vStride, _padding, _activationFunction, _name);
        }
        /// <summary>
        /// Adds a convolution layer for a two-dimensional vector. If the previous layer has a non-two-dimensional output, an exception is thrown
        /// </summary>
        /// <param name="kernelWidth">Convolution core width (columns in a two-dimensional matrix)</param>
        /// <param name="kernelHeight">Convolution core height (rows in a two-dimensional matrix)</param>
        /// <param name="outFeatureMapCount">Bit depth of the output cell after convolution</param>
        /// <param name="activationFunction">Activation function for the output layer. If not required, pass null</param>
        /// <param name="hStride">The step of moving down the convolution window horizontally (along the matrix columns)</param>
        /// <param name="vStride">Step of shifting the convolution window vertically (along the rows of the matrix)</param>
        /// <param name="padding">Fill when using convolution</param>
        /// <param name="name"></param>
        public Convolution2D(int kernelWidth, int kernelHeight, int outFeatureMapCount = 1, int hStride = 1, int vStride = 1, Padding padding = Padding.Valid, ActivationFunction activationFunction = null, string name = "Conv2D")
        {
            _kernelWidth = kernelWidth;
            _kernelHeight = kernelHeight;
            _outFeatureMapCount = outFeatureMapCount;
            _hStride = hStride;
            _vStride = vStride;
            _padding = padding;
            _activationFunction = activationFunction;
            _name = name;
        }
        public override string GetDescription()
        {
            return $"Conv2D(K={_kernelWidth}x{_kernelHeight}S={_hStride}x{_vStride}P={_padding})[{_activationFunction?.GetDescription()}]";
        }
    }
    /// <summary>
    /// Specifies padding when using convolution
    /// </summary>
    public enum Padding
    {
        /// <summary>
        /// There is no filling of the edges (convolution core movement is strictly limited by the image size), the image is minimized according to the classics: n-f + 1 x n-f + 1
        /// </summary>
        Valid,
        /// <summary>
        /// There are fillings of the edges (the convolution core moves beyond the boundaries of the image, the excess is supplemented with zeros, the output image remains the same size as before the convolution), the image is minimized: n + 2p-f + 1 x n + 2p-f + 1
        /// </summary>
        Same
    }
}
