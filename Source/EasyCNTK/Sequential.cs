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
using EasyCNTK.Layers;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace EasyCNTK
{
    /// <summary>
    /// Implements the operations of constructing a direct distribution model
    /// </summary>
    /// <typeparam name = "T"> Data type. <Seealso cref = "float" />, <seealso cref = "double" /> </typeparam> supported
    public sealed class Sequential<T> : IDisposable where T : IConvertible
    {
        public const string PrefixFilenameDescription = "ArchitectureDescription";
        private DeviceDescriptor Device { get; }
        private string _architectureDescription;

        private readonly Dictionary<string, Function> _shortcutConnectionInputs = new Dictionary<string, Function>();

        private string GetArchitectureDescription()
        {
            var shortcuts = _shortcutConnectionInputs.Keys.ToList();
            foreach (var shortcut in shortcuts)
            {
                if (!_architectureDescription.Contains($"ShortOut({shortcut})"))
                {
                    _architectureDescription = _architectureDescription.Replace($"-ShortIn({shortcut})", "");
                }
            }
            return _architectureDescription + "[OUT]";
        }

        /// <summary>
        /// Loads the model from the file. Also trying to read the description of the network architecture:
        /// 1) From the ArchitectureDescription file {model_file_name} .txt
        /// 2) From the model file name, focusing on the presence of [IN] and [OUT] tags. If this fails, then the configuration description: Unknown.
        /// </summary>
        /// <typeparam name = "T"> Data type. <Seealso cref = "float" />, <seealso cref = "double" /> </typeparam> supported
        /// <param name = "device"> Device to boot </param>
        /// <param name = "filePath"> Path to the model file </param>
        /// <param name = "modelFormat"> Model format </param>
        /// <returns></returns>
        public static Sequential<T> LoadModel(DeviceDescriptor device, string filePath, ModelFormat modelFormat = ModelFormat.CNTKv2)
        {
            return new Sequential<T>(device, filePath, modelFormat);
        }

        /// <summary>
        /// Initializes the neural network with the dimension of the input vector without layers
        /// </summary>
        /// <param name = "inputShape"> Tensor describing the input form of the neural network (input) </param>
        /// <param name = "device"> The device on which the network is being created </param>
        /// <param name = "inputDynamicAxes"> List of dynamic axes. Add the axis <seealso cref = "Axis.DefaultBatchAxis ()" /> if the output of your network is a sequence. </Param>
        /// <param name = "isSparse"> Indicates that the input is a One-Hot-Encoding vector and you should use the CNTK internal optimization to increase performance. </param>
        public Sequential(DeviceDescriptor device, int[] inputShape, string inputName = "", IList<Axis> inputDynamicAxes = null, bool isSparse = false)
        {
            Device = device;
            var dataType = typeof(T) == typeof(double) ? DataType.Double : DataType.Float;
            Model = Variable.InputVariable(inputShape, dataType, string.IsNullOrEmpty(inputName) ? "Input" : inputName, inputDynamicAxes, isSparse);
            var shape = "";
            inputShape.ToList().ForEach(p =>
            {
                shape += p.ToString() + "x";
            });
            shape = shape.Substring(0, shape.Length - 1);
            _architectureDescription = $"[IN]{shape}";
        }

        public Sequential(DeviceDescriptor device, Variable input)
        {
            Device = device;
            Model = input;
            var shape = "";
            input.Shape.Dimensions.ToList().ForEach(p =>
            {
                shape += p.ToString() + "x";
            });
            shape = shape.Substring(0, shape.Length - 1);
            _architectureDescription = $"[IN]{shape}";
        }

        private Sequential(DeviceDescriptor device, string filePath, ModelFormat modelFormat = ModelFormat.CNTKv2)
        {
            Device = device;
            Model = Function.Load(filePath, device, modelFormat);
            var dataType = typeof(T) == typeof(double) ? DataType.Double : DataType.Float;
            if (Model.Output.DataType != dataType)
            {
                throw new ArgumentException($"The universal parameter TElement does not match the data type in the model. Required type: {Model.Output.DataType}");
            }
            try
            {
                _architectureDescription = "Unknown";

                var pathToFolder = Directory.GetParent(filePath).FullName;
                var descriptionPath = Path.Combine(pathToFolder, $"{PrefixFilenameDescription}_{filePath}.txt");
                if (File.Exists(descriptionPath))
                {
                    var description = File.ReadAllText(descriptionPath);
                    var index = description.IndexOf("[OUT]", StringComparison.Ordinal);
                    _architectureDescription = index != -1 ? description.Remove(index) : description;
                    return;
                }

                var fileName = Path.GetFileName(filePath);
                var indexIn = fileName.IndexOf("[IN]", StringComparison.Ordinal);
                var indexOut = fileName.IndexOf("[OUT]", StringComparison.Ordinal);
                var fileNameContainsArchitectureDescription = indexIn != -1 && indexOut != -1 && indexIn < indexOut;
                if (fileNameContainsArchitectureDescription)
                {
                    _architectureDescription = fileName.Substring(indexIn, indexOut - indexIn);
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error loading model configuration file. {ex.Message}");
            }
        }

        /// <summary>
        /// Adds the specified layer (joins to the last added layer)
        /// </summary>
        /// <param name = "layer"> Layer for docking </param>
        public void Add(Layer layer)
        {
            Model = layer.Create(Model, Device);
            _architectureDescription += $"-{layer.GetDescription()}";
        }

        /// <summary>
        /// Creates an entry point for SC, from which you can create a connection to the next network layers. At least one output point must exist for one input point, otherwise the connection is ignored in the model.
        /// </summary>
        /// <param name = "nameShortcutConnection"> The name of the entry point from which the connection will be forwarded. Within the network must be unique </param>
        public void CreateInputPointForShortcutConnection(string nameShortcutConnection)
        {
            _shortcutConnectionInputs.Add(nameShortcutConnection, Model);
            _architectureDescription += $"-ShortIn({nameShortcutConnection})";
        }

        /// <summary>
        /// Creates an output point for SC, to which the connection is forwarded from a previously created input point. For one input point there may be several output points.
        /// </summary>
        /// <param name = "nameShortcutConnection"> The name of the entry point from which the connection is forwarded. </param>
        public void CreateOutputPointForShortcutConnection(string nameShortcutConnection)
        {
            if (_shortcutConnectionInputs.TryGetValue(nameShortcutConnection, out var input))
            {
                if (input.Output.Shape.Equals(Model.Output.Shape))
                {
                    Model = CNTKLib.Plus(Model, input);
                }
                else if (input.Output.Shape.Rank != 1 && Model.Output.Shape.Rank == 1) // [3x4x2] => [5]
                {
                    int targetDim = Model.Output.Shape[0];
                    int inputDim = input.Output.Shape.Dimensions.Aggregate((d1, d2) => d1 * d2);
                    var inputVector = CNTKLib.Reshape(input, new[] { inputDim });

                    var scale = new Parameter(new[] { targetDim, inputDim }, input.Output.DataType, CNTKLib.UniformInitializer(CNTKLib.DefaultParamInitScale), Device);
                    var scaled = CNTKLib.Times(scale, inputVector);

                    var reshaped = CNTKLib.Reshape(scaled, Model.Output.Shape);
                    Model = CNTKLib.Plus(reshaped, Model);
                }
                else if (input.Output.Shape.Rank == 1 && Model.Output.Shape.Rank != 1) // [5] => [3x4x2]
                {
                    int targetDim = Model.Output.Shape.Dimensions.Aggregate((d1, d2) => d1 * d2);
                    var inputDim = input.Output.Shape[0];
                    var scale = new Parameter(new[] { targetDim, inputDim }, input.Output.DataType, CNTKLib.UniformInitializer(CNTKLib.DefaultParamInitScale), Device);
                    var scaled = CNTKLib.Times(scale, input);

                    var reshaped = CNTKLib.Reshape(scaled, Model.Output.Shape);
                    Model = CNTKLib.Plus(reshaped, Model);
                }
                else // [3x4x2] => [4x5x1] || [3x1] => [5x7x8x1]
                {
                    var inputDim = input.Output.Shape.Dimensions.Aggregate((d1, d2) => d1 * d2);
                    var inputVector = CNTKLib.Reshape(input, new[] { inputDim });

                    var targetDim = Model.Output.Shape.Dimensions.Aggregate((d1, d2) => d1 * d2);
                    var scale = new Parameter(new[] { targetDim, inputDim }, input.Output.DataType, CNTKLib.UniformInitializer(CNTKLib.DefaultParamInitScale), Device);
                    var scaled = CNTKLib.Times(scale, inputVector);

                    var reshaped = CNTKLib.Reshape(scaled, Model.Output.Shape);
                    Model = CNTKLib.Plus(reshaped, Model);
                }
                _architectureDescription += $"-ShortOut({nameShortcutConnection})";
            }
        }

        /// <summary>
        /// Configured CNTK model
        /// </summary>
        public Function Model { get; private set; }

        /// <summary>
        /// Saves the model to a file.
        /// </summary>
        /// <param name = "filePath"> Path to save the model (including the file name and extension) </param>
        /// <param name = "saveArchitectureDescription"> Specifies whether the architecture description should be saved in a separate file: ArchitectureDescription_ {model-file-name} .txt </param>
        public void SaveModel(string filePath, bool saveArchitectureDescription = true)
        {
            Model.Save(filePath);
            if (saveArchitectureDescription)
            {
                var fileName = Path.GetFileName(filePath);
                var pathToFolder = Directory.GetParent(filePath).FullName;
                var descriptionPath = Path.Combine(pathToFolder, $"{PrefixFilenameDescription}_{fileName}.txt");
                using (var stream = File.CreateText(descriptionPath))
                {
                    stream.Write(GetArchitectureDescription());
                }
            }
        }

        public override string ToString()
        {
            return GetArchitectureDescription();
        }

        public void Dispose()
        {
            Model.Dispose();
            Device.Dispose();
            foreach (var shortcut in _shortcutConnectionInputs.Values)
            {
                shortcut.Dispose();
            }
        }
    }
}

