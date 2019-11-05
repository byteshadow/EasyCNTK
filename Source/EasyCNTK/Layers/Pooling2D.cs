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
    /// Implements a pulling layer for a two-dimensional vector
    /// </summary>
    public sealed class Pooling2D : Layer
    {
        private int _poolingWindowWidth;
        private int _poolingWindowHeight;
        private int _hStride;
        private int _vStride;
        private PoolingType _poolingType;
        private string _name;
        /// <summary>
        /// Adds a pulling layer for a two-dimensional vector. If the previous layer has a non-two-dimensional output, an exception is thrown
        /// </summary>
        /// <param name="poolingWindowWidth">Pulling window width</param>
        /// <param name="poolingWindowHeight">Pulling window height</param>
        /// <param name="hStride">Horizontal displacement step of the pulling window (according to matrix columns)</param>
        /// <param name="vStride">Step of shifting the pulling window vertically (along the rows of the matrix)</param>
        /// <param name="poolingType">Type of pulling. Maximum or medium</param>
        /// <param name="name"></param>
        public static Function Build(Variable input, int poolingWindowWidth, int poolingWindowHeight, int hStride, int vStride, PoolingType poolingType, string name)
        {
            var pooling = CNTKLib.Pooling(input, poolingType, new int[] { poolingWindowWidth, poolingWindowHeight }, new int[] { hStride, vStride }, new bool[] { true });
            return CNTKLib.Alias(pooling, name);
        }
        public override Function Create(Function input, DeviceDescriptor device)
        {
            return Build(input, _poolingWindowWidth, _poolingWindowHeight, _hStride, _vStride, _poolingType, _name);
        }
        /// <summary>
        /// Creates a pulling layer for a two-dimensional vector. If the previous layer has a non-two-dimensional output, an exception is thrown
        /// </summary>
        /// <param name="poolingWindowWidth">Pulling window width</param>
        /// <param name="poolingWindowHeight">Pulling window height</param>
        /// <param name="hStride">Horizontal displacement step of the pulling window (according to matrix columns)</param>
        /// <param name="vStride">Step of shifting the pulling window vertically (along the rows of the matrix)</param>
        /// <param name="poolingType">Type of pulling. Maximum or medium</param>
        /// <param name="name"></param>
        public Pooling2D(int poolingWindowWidth, int poolingWindowHeight, int hStride, int vStride, PoolingType poolingType, string name = "Pooling2D")
        {
            _poolingWindowWidth = poolingWindowWidth;
            _poolingWindowHeight = poolingWindowHeight;
            _hStride = hStride;
            _vStride = vStride;
            _poolingType = poolingType;
            _name = name;
        }        
        public override string GetDescription()
        {
            return $"Pooling2D(W={_poolingWindowWidth}x{_poolingWindowHeight}S={_hStride}x{_vStride}T={_poolingType})";
        }
    }
}
