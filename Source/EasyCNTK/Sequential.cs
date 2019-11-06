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
    /// It implements the operations of constructing a direct distribution model with one input and one output
    /// </summary>
    /// <typeparam name="T">Data type. Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
    public sealed class Sequential<T> : IDisposable where T : IConvertible
    {
        public const string PREFIX_FILENAME_DESCRIPTION = "ArchitectureDescription";
        private DeviceDescriptor _device;
        private string _architectureDescription;
        private Dictionary<string, Function> _shortcutConnectionInputs = new Dictionary<string, Function>();

        private string getArchitectureDescription()
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
        /// Loads a model from a file. Also trying to read the description of the network architecture: 
        /// 1) From the file ArchitectureDescription {model_file_name} .txt 
        /// 2) From the model file name, focusing on the presence of [IN] and [OUT] tags. If this fails, then the configuration description is: Unknown.
        /// </summary>
        /// <typeparam name="T">Data type. Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="device">Boot device</param>
        /// <param name="filePath">Model file path</param>
        /// <param name="modelFormat">Model Format</param>
        /// <returns></returns>
        public static Sequential<T> LoadModel(DeviceDescriptor device, string filePath, ModelFormat modelFormat = ModelFormat.CNTKv2)
        {
            return new Sequential<T>(device, filePath, modelFormat);
        }

        /// <summary>
        /// Initializes a neural network with the dimension of the input vector without layers
        /// </summary>
        /// <param name="inputShape">Tensor describing the input form of the neural network (input data)</param>
        /// <param name="device">The device on which the network is created</param>
        /// <param name="outputIsSequence">Indicates that the network output is a sequence.</param>
        /// <param name="inputName">Neural Network Login</param>
        /// <param name="isSparce">Indicates that the input is a One-Hot-Encoding vector and that CNTK&#39;s internal optimization should be used to increase performance.</param>
        public Sequential(DeviceDescriptor device, int[] inputShape, bool outputIsSequence = false, string inputName = "Input", bool isSparce = false)
        {
            _device = device;
            var dataType = typeof(T) == typeof(double) ? DataType.Double : DataType.Float;
            Model = outputIsSequence
                ? Variable.InputVariable(inputShape, dataType, inputName, new[] { Axis.DefaultBatchAxis() }, isSparce)
                : Variable.InputVariable(inputShape, dataType, inputName, null, isSparce);
            var shape = "";
            inputShape.ToList().ForEach(p =>
            {
                shape += p.ToString() + "x";
            });
            shape = shape.Substring(0, shape.Length - 1);
            _architectureDescription = $"[IN]{shape}";
        }

        private Sequential(DeviceDescriptor device, string filePath, ModelFormat modelFormat = ModelFormat.CNTKv2)
        {
            _device = device;
            Model = Function.Load(filePath, device, modelFormat);
            var dataType = typeof(T) == typeof(double) ? DataType.Double : DataType.Float;
            if (Model.Output.DataType != dataType)
            {
                throw new ArgumentException($"Universal parameter {nameof(T)} does not match the data type in the model. Type required: {Model.Output.DataType}");
            }
            try
            {
                _architectureDescription = "Unknown";

                var pathToFolder = Directory.GetParent(filePath).FullName;
                var descriptionPath = Path.Combine(pathToFolder, $"{PREFIX_FILENAME_DESCRIPTION}_{filePath}.txt");
                if (File.Exists(descriptionPath))
                {
                    var description = File.ReadAllText(descriptionPath);
                    var index = description.IndexOf("[OUT]");
                    _architectureDescription = index != -1 ? description.Remove(index) : description;
                    return;
                }

                var fileName = Path.GetFileName(filePath);
                var indexIn = fileName.IndexOf("[IN]");
                var indexOut = fileName.IndexOf("[OUT]");
                bool fileNameContainsArchitectureDescription = indexIn != -1 && indexOut != -1 && indexIn < indexOut;
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
        /// Adds the specified layer (joins the last added layer)
        /// </summary>
        /// <param name="layer">Docking layer</param>
        public void Add(Layer layer)
        {
            Model = layer.Create(Model, _device);
            _architectureDescription += $"-{layer.GetDescription()}";
        }
        /// <summary>
        /// Creates an entry point for SC from which you can create a connection to the following layers of the network. At least one output point must exist for one input point, otherwise the connection is ignored in the model.
        /// </summary>
        /// <param name="nameShortcutConnection">The name of the entry point from which the connection will be forwarded. Within the network must be unique</param>
        public void CreateInputPointForShortcutConnection(string nameShortcutConnection)
        {
            _shortcutConnectionInputs.Add(nameShortcutConnection, Model);
            _architectureDescription += $"-ShortIn({nameShortcutConnection})";
        }
        /// <summary>
        /// Creates an exit point for SC to which a connection is forwarded from a previously created entry point. For one input point, several output points can exist.
        /// </summary>
        /// <param name="nameShortcutConnection">The name of the entry point from which the connection is forwarded.</param>
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

                    var scale = new Parameter(new[] { targetDim, inputDim }, input.Output.DataType, CNTKLib.UniformInitializer(CNTKLib.DefaultParamInitScale), _device);
                    var scaled = CNTKLib.Times(scale, inputVector);

                    var reshaped = CNTKLib.Reshape(scaled, Model.Output.Shape);
                    Model = CNTKLib.Plus(reshaped, Model);
                }
                else if (input.Output.Shape.Rank == 1 && Model.Output.Shape.Rank != 1) // [5] => [3x4x2]
                {
                    int targetDim = Model.Output.Shape.Dimensions.Aggregate((d1, d2) => d1 * d2);
                    var inputDim = input.Output.Shape[0];
                    var scale = new Parameter(new[] { targetDim, inputDim }, input.Output.DataType, CNTKLib.UniformInitializer(CNTKLib.DefaultParamInitScale), _device);
                    var scaled = CNTKLib.Times(scale, input);

                    var reshaped = CNTKLib.Reshape(scaled, Model.Output.Shape);
                    Model = CNTKLib.Plus(reshaped, Model);
                }
                else // [3x4x2] => [4x5x1] || [3x1] => [5x7x8x1]
                {
                    var inputDim = input.Output.Shape.Dimensions.Aggregate((d1, d2) => d1 * d2);
                    var inputVector = CNTKLib.Reshape(input, new[] { inputDim });

                    var targetDim = Model.Output.Shape.Dimensions.Aggregate((d1, d2) => d1 * d2);
                    var scale = new Parameter(new[] { targetDim, inputDim }, input.Output.DataType, CNTKLib.UniformInitializer(CNTKLib.DefaultParamInitScale), _device);
                    var scaled = CNTKLib.Times(scale, inputVector);

                    var reshaped = CNTKLib.Reshape(scaled, Model.Output.Shape);
                    Model = CNTKLib.Plus(reshaped, Model);
                }
                _architectureDescription += $"-ShortOut({nameShortcutConnection})";
            }
        }
        /// <summary>
        /// Configured CNTK Model
        /// </summary>
        public Function Model { get; private set; }
        /// <summary>
        /// Saves the model to a file.
        /// </summary>
        /// <param name="filePath">Path to save the model (including file name and extension)</param>
        /// <param name="saveArchitectureDescription">Specifies whether to save the architecture description in a separate file: ArchitectureDescription_ {model-filename} .txt</param>
        public void SaveModel(string filePath, bool saveArchitectureDescription = true)
        {
            Model.Save(filePath);
            if (saveArchitectureDescription)
            {
                var fileName = Path.GetFileName(filePath);
                var pathToFolder = Directory.GetParent(filePath).FullName;
                var descriptionPath = Path.Combine(pathToFolder, $"{PREFIX_FILENAME_DESCRIPTION}_{fileName}.txt");
                using (var stream = File.CreateText(descriptionPath))
                {
                    stream.Write(getArchitectureDescription());
                }
            }
        }

        public override string ToString()
        {
            return getArchitectureDescription();
        }

        #region IDisposable Support
        private bool disposedValue = false; // To identify redundant calls

        void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    // TODO: release managed state (managed objects).
                    Model.Dispose();
                    _device.Dispose();
                    foreach (var shortcut in _shortcutConnectionInputs.Values)
                    {
                        shortcut.Dispose();
                    }
                }

                // TODO: Release unmanaged resources (unmanaged objects) and override the completion method below.
                // TODO: set large fields to NULL.
                Model = null;
                _device = null;
                _shortcutConnectionInputs = null;
                disposedValue = true;
            }
        }

        // TODO: Override the completion method only if Dispose (bool disposing) above includes code to free unmanaged resources.
        ~Sequential()
        {
            // Do not modify this code. Place the cleanup code above in the Dispose (bool disposing) method.
            Dispose(false);
        }

        // This code has been added to properly implement the released class template.
        public void Dispose()
        {
            // Do not modify this code. Place the cleanup code above in the Dispose (bool disposing) method.
            Dispose(true);
            // TODO: uncomment the next line if the completion method is overridden above.
            GC.SuppressFinalize(this);
        }
        #endregion

    }
}

