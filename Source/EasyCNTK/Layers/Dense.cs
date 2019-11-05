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
    /// Implements a fully connected layer with a given activation function
    /// </summary>
    public sealed class Dense: Layer
    {
        private int _outputDim;
        private ActivationFunction _activationFunction;
        private string _name;

        /// <summary>
        /// Creates a fully connected layer with the specified activation function.
        /// </summary>
        /// <param name="input">Input variable (layer) of a given bit depth</param>
        /// <param name="outputDim">Output capacity (number of neurons)</param>
        /// <param name="activationFunction">Activation function</param>
        /// <param name="device">The device on which the calculation is made</param>
        /// <param name="name">Layer name</param>
        /// <returns></returns>
        private static Function createFullyConnectedLinearLayer(Variable input, int outputDim, ActivationFunction activationFunction, DeviceDescriptor device, string name)
        {
            var dataType = input.DataType;            
            var inputDim = input.Shape[0];
            var weight   = new Parameter(new int[] { outputDim, inputDim }, dataType, CNTKLib.GlorotUniformInitializer(
                CNTKLib.DefaultParamInitScale,
                    CNTKLib.SentinelValueForInferParamInitRank,
                    CNTKLib.SentinelValueForInferParamInitRank, 1), device);
            var bias                    = new Parameter(new int[] { outputDim }, dataType, 0, device);
            var fullyConnected          = CNTKLib.Times(weight, input) + bias;
            var activatedFullyConnected = activationFunction?.ApplyActivationFunction(fullyConnected, device) ?? fullyConnected;
            return Function.Alias(activatedFullyConnected, name);
        }
        /// <summary>
        /// Creates a fully connected layer with the specified activation function.
        /// </summary>
        /// <param name="input">Input variable (layer) of a given bit depth</param>
        /// <param name="outputDim">Output capacity (number of neurons)</param>
        /// <param name="activationFunction">Activation function</param>
        /// <param name="device">The device on which the calculation is made</param>
        /// <param name="name">Layer name</param>
        /// <returns></returns>
        public static Function Build(Function input, int outputDim, ActivationFunction activationFunction, DeviceDescriptor device, string name = "Dense")
        {
            return createFullyConnectedLinearLayer(input, outputDim, activationFunction, device, name);
        }        
        public override Function Create(Function input, DeviceDescriptor device)
        {
            return createFullyConnectedLinearLayer(input, _outputDim, _activationFunction, device, _name);
        }
        /// <summary>
        /// Creates a fully connected layer with the specified activation function.
        /// </summary>
        /// <param name="outputDimension">Output capacity (number of neurons)</param>
        /// <param name="activationFunction">Activation function, null if not required</param>
        /// <param name="name">Layer name</param>
        public Dense(int outputDimension, ActivationFunction activationFunction, string name = "Dense")
        {
            _outputDim = outputDimension;
            _activationFunction = activationFunction;
            _name = name;
        }  

        public override string GetDescription()
        {
            return $"{_outputDim}[{_activationFunction?.GetDescription()}]";
        }
    }
}
