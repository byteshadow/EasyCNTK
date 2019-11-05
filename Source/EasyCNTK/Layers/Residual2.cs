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
    /// Implements a residual layer with 2 inner layers
    /// </summary>
    public sealed class Residual2 : Layer
    {
        private int _outputDim;
        private ActivationFunction _activationFunction;
        private string _name;

        private static Function createResidualLayer2(Function input, int outputDimension, ActivationFunction activationFunction, DeviceDescriptor device, string name)
        {
            var dataType = input.Output.DataType;
                          
            //forwarding of the entrance past 1 layer    
            var forwarding = input;
            if (outputDimension != input.Output.Shape[0])
            {
                var scales = new Parameter(new int[] { outputDimension, input.Output.Shape[0] }, dataType, CNTKLib.UniformInitializer(CNTKLib.DefaultParamInitScale), device);
                forwarding = CNTKLib.Times(scales, forwarding);
            }

            //create 1 layer
            input = Dense.Build(input, outputDimension, activationFunction, device, "");
            //creation of 2 layers without activation function
            input = Dense.Build(input, outputDimension, null, device, "");
            //connection with forwarded input
            input = CNTKLib.Plus(input, forwarding);

            input = activationFunction?.ApplyActivationFunction(input, device) ?? input;
            return Function.Alias(input, name);
        }
        /// <summary>
        /// Creates a residual layer with 2 inner layers
        /// </summary>        
        /// <param name="input">entrance</param>
        /// <param name="outputDimension">Output layer depth</param>
        /// <param name="activationFunction">Activation function, null if not required</param>
        /// <param name="device">Device for calculations</param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Function Build(Function input, int outputDimension, ActivationFunction activationFunction, DeviceDescriptor device, string name = "Residual2")
        {
            return createResidualLayer2(input, outputDimension, activationFunction, device, name);
        }
        /// <summary>
        /// Creates a residual layer with 2 inner layers
        /// </summary>
        /// <param name="outputDimension">Output layer depth</param>
        /// <param name="activationFunction">Activation function, null if not required</param>
        /// <param name="name"></param>
        public Residual2(int outputDimension, ActivationFunction activationFunction, string name = "Residual2")
        {
            _outputDim = outputDimension;
            _activationFunction = activationFunction;
            _name = name;
        }
        public override Function Create(Function input, DeviceDescriptor device)
        {
            return createResidualLayer2(input, _outputDim, _activationFunction, device, _name);
        }
     
        public override string GetDescription()
        {
            return $"Res2({_outputDim})[{_activationFunction?.GetDescription()}]";
        }
    }
}
