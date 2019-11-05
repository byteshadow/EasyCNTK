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
using System.Text;

namespace EasyCNTK
{
    /// <summary>
    /// It implements the operations of constructing a direct distribution model with one input and several outputs
    /// </summary>
    /// <typeparam name="T">Data type. Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
    public sealed class SequentialMultiOutput<T>: IDisposable where T:IConvertible
    {
        class Branch
        {
            public int Index { get; set; }
            public string Name { get; set; }
            public string ArchitectureDescription { get; set; }
            public Function Model { get; set; }
        }

        public const string PREFIX_FILENAME_DESCRIPTION = "ArchitectureDescription";
        private DeviceDescriptor _device;
        private Function _model;
        private string _architectureDescription;
        private Dictionary<string, Branch> _branches = new Dictionary<string, Branch>();        
        private bool _isCompiled = false;
        private string getArchitectureDescription()
        {
            var descriptionBranches = _branches
                .Values
                .OrderBy(p => p.Index)
                .Select(p => p.ArchitectureDescription);
            StringBuilder stringBuilder = new StringBuilder(_architectureDescription);
            foreach (var branch in descriptionBranches)
            {
                stringBuilder.Append(branch);
                stringBuilder.Append("[OUT]");
            }
            return stringBuilder.ToString();
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
        public static SequentialMultiOutput<T> LoadModel(DeviceDescriptor device, string filePath, ModelFormat modelFormat = ModelFormat.CNTKv2)
        {
            return new SequentialMultiOutput<T>(device, filePath, modelFormat);
        }
        /// <summary>
        /// Initializes a neural network with the dimension of the input vector without layers
        /// </summary>
        /// <param name="inputShape">Tensor describing the input form of the neural network (input data)</param>
        /// <param name="device">The device on which the network is created</param>
        /// <param name="outputIsSequence">Indicates that the network output is a sequence.</param>
        /// <param name="inputName">Neural Network Login</param>
        /// <param name="isSparce">Indicates that the input is a One-Hot-Encoding vector and that CNTK&#39;s internal optimization should be used to increase performance.</param>
        public SequentialMultiOutput(DeviceDescriptor device, int[] inputShape, bool outputIsSequence = false, string inputName = "Input", bool isSparce = false)
        {
            _device = device;
            var dataType = typeof(T) == typeof(double) ? DataType.Double : DataType.Float;
            _model = outputIsSequence
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

        private SequentialMultiOutput(DeviceDescriptor device, string filePath, ModelFormat modelFormat = ModelFormat.CNTKv2)
        {
            _isCompiled = true;
            _device = device;
            _model = Function.Load(filePath, device, modelFormat);
            var dataType = typeof(T) == typeof(double) ? DataType.Double : DataType.Float;
            if (_model.Output.DataType != dataType)
            {
                throw new ArgumentException($"Универсальный параметр {nameof(T)} не сответствует типу данных в модели. Требуемый тип: {_model.Output.DataType}");
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
                var indexOut = fileName.LastIndexOf("[OUT]");
                bool fileNameContainsArchitectureDescription = indexIn != -1 && indexOut != -1 && indexIn < indexOut;
                if (fileNameContainsArchitectureDescription)
                {
                    _architectureDescription = fileName.Substring(indexIn, indexOut - indexIn);
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Ошибка загрузки файла конфигурации модели. {ex.Message}");
            }
        }
        /// <summary>
        /// Adds the specified layer (joins the last added layer)
        /// </summary>
        /// <param name="layer">Docking layer</param>
        public void Add(Layer layer)
        {
            if (_isCompiled)
                throw new NotSupportedException("Изменение скомпилированной модели не поддерживается.");
            _model = layer.Create(_model, _device);
            _architectureDescription += $"-{layer.GetDescription()}";
        }
        /// <summary>
        /// Adds the specified layer to the specified branch (joins the last added layer of the branch)
        /// </summary>
        /// <param name="branch">The name of the branch. Must match one of the names specified during the call<seealso cref="SplitToBranches(string[])"/></param>
        /// <param name="layer">Docking layer</param>
        public void AddToBranch(string branch, Layer layer)
        {
            if (_isCompiled)
                throw new NotSupportedException("Изменение скомпилированной модели не поддерживается.");
            if (_branches.Count == 0)
                throw new NotSupportedException("Добавления слоя к ветви без предварительного создания ветвей не поддерживается, сначала создайте ветви методом SplitToBranches()");
            if (_branches.TryGetValue(branch, out var branchOutput))
            {
                _branches[branch].Model = layer.Create(branchOutput.Model, _device);
                _branches[branch].ArchitectureDescription += $"-{layer.GetDescription()}";                
            }
            else
            {
                throw new ArgumentException($"Ветви с именем '{branch}' не существует.", nameof(branch));
            }
        }
        /// <summary>
        /// Splits the main sequence of layers into several branches
        /// </summary>
        /// <param name="branchNames">The names of the branches, each branch in the order of listing will be associated with the corresponding output of the network. Names must be unique.</param>
        public void SplitToBranches(params string[] branchNames)
        {
            if (_isCompiled)
                throw new NotSupportedException("Изменение скомпилированной модели не поддерживается.");
            if (_branches.Count != 0)
                throw new NotSupportedException("Повторное разбиение сущеcтвующих ветвей на новые ветви не поддерживается.");
            if (branchNames.Length < 2)
                throw new NotSupportedException("Разбиение возможно минимум на 2 ветви.");
            _branches = branchNames
                .Select((branch, index) => (branch, index, _model))
                .ToDictionary(p => p.branch, q => new Branch()
                {
                    Index = q.index,
                    Name = q.branch,
                    ArchitectureDescription = $"-#{q.branch}",
                    Model = _model
                });                      
        }
        /// <summary>
        /// Compiles all created branches into one model
        /// </summary>
        public void Compile()
        {
            var outputs = _branches
                .Values
                .OrderBy(p => p.Index)
                .Select(p => (Variable)p.Model)
                .ToList();
            _model = CNTKLib.Combine(new VariableVector(outputs));
            
            _isCompiled = true;
        }

        //private bool _isDisposed = false;
        //public void Dispose()
        //{
        //    if (!_isDisposed)
        //    {
        //        _device.Dispose();
        //        _model.Dispose();
        //        foreach (var item in _branches.Values)
        //        {
        //            item.Model.Dispose();
        //        }
        //    }
        //    _isDisposed = true;
        //}
        /// <summary>
        /// Compiled CNTK Model 
        /// </summary>
        public Function Model
        {
            get
            {
                if (!_isCompiled)
                    throw new NotSupportedException("Использование нескомпилированной модели не поддерживается. Скомпилируйте вызвав Compile()");
                return _model;
            }
        }
        public override string ToString()
        {
            return getArchitectureDescription();
        }
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

        #region IDisposable Support
        private bool disposedValue = false; // To identify redundant calls

        void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    // TODO: release managed state (managed objects).
                    _device.Dispose();
                    _model.Dispose();
                    foreach (var item in _branches.Values)
                    {
                        item.Model.Dispose();
                    }
                }

                // TODO: Release unmanaged resources (unmanaged objects) and override the completion method below.
                // TODO: set large fields to NULL.
                _device = null;
                _model = null;
                _branches = null;

                disposedValue = true;
            }
        }

        // TODO: Override the completion method only if Dispose (bool disposing) above includes code to free unmanaged resources.
        ~SequentialMultiOutput()
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
