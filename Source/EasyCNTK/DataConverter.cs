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
using System;
using System.Collections.Generic;
using System.Linq;

namespace EasyCNTK
{
    /// <summary>
    /// Implements methods for converting native data into a format suitable for training at CNTK
    /// </summary>
    public class DataConverter:IDisposable
    {
        protected DeviceDescriptor Device { get; set; }
        /// <summary>
        /// Splits an input sequence into segments (subsequences) of equal size
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="source">Source sequence</param>
        /// <param name="segmentSize">Segment size (number of elements)</param>
        /// <returns></returns>
        protected IEnumerable<IList<T>> GetSegments<T>(IEnumerable<T> source, int segmentSize)
        {
            IList<T> list = new List<T>(segmentSize);
            foreach (var item in source)
            {
                list.Add(item);
                if (list.Count == segmentSize)
                {
                    yield return list;
                    list = new List<T>(segmentSize);
                }
            }
            if (list.Count > 0)
            {
                yield return list;
            }
        }
        protected T[] MatrixToVector<T>(T[,] matrix)
        {
            var rows = matrix.GetLength(0);
            var columns = matrix.GetLength(1);
            var result = new T[rows * columns];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    result[(i + 1) * j] = matrix[i, j];
                }
            }
            return result;
        }
        protected int GetRowsCount<T>(T[,] matrix)
        {
            return matrix.GetLength(0);
        }
        protected int GetColumnsCount<T>(T[,] matrix)
        {
            return matrix.GetLength(1);
        }
        /// <summary>
        /// Initializes the converter to work with the specified device (CPU, GPU)
        /// </summary>
        /// <param name="device">Device for calculations</param>
        public DataConverter(DeviceDescriptor device)
        {            
            Device = device ?? throw new ArgumentNullException(nameof(device));
        }
        /// <summary>
        /// Converts a dataset into sets of training examples for use in recursive networks. 
        /// </summary>
        /// <typeparam name="T">Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="features">A set of sequences (traits). Each sequence can be of variable length, but of the same dimension (the arrays of which the sequence consists must have the same length)</param>
        /// <param name="labels">Set of labels. The dimension of the labels should be the same.</param>
        /// <param name="minibatchSize">Minipack size</param>
        /// <returns></returns>
        public IEnumerable<Minibatch> ConvertDatasetToMinibatch<T>(IEnumerable<IList<T[]>> features, IEnumerable<T[]> labels, int minibatchSize) where T:IConvertible
        {
            var inputDimension = features.FirstOrDefault()?[0].Length ?? 0;
            var outputDimension = labels.FirstOrDefault()?.Length ?? 0;
            var combined = features.Zip(labels, (f, l) => (f, l));
            foreach (var batchData in GetSegments(combined, minibatchSize))
            {
                //{}-miniBatch, ()-sequence, []-features.  
                //{ ([inputDim], [inputDim], [inputDim]), ([inputDim], [inputDim]) } => { [inputDim * 3], [inputDim * 2] }
                var featuresTransformed = batchData.Select(p => p.f.SelectMany(q => q));
                //{ [outputDim], [outputDim] } => { outputDim * 2 }
                var labelTransformed = batchData.SelectMany(p => p.l);

                Minibatch minibatch = new Minibatch();
                minibatch.Features = Value.CreateBatchOfSequences(new[] { inputDimension }, featuresTransformed, Device);
                minibatch.Labels = Value.CreateBatch(new[] { outputDimension }, labelTransformed, Device);
                minibatch.Size = batchData.Count;
                yield return minibatch;
            }
        }

        /// <summary>
        /// Converts a dataset into sets of training examples for use in CNTK. 
        /// </summary>
        /// <typeparam name="T">Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="dataset">Dataset. Each example should contain signs at the beginning of the array with dimensions of inputDim, and at the end of class labels with dimensions of outputDim. 
        /// For example, inputDim = 3, outputDim = 2: [f1, f2, f3, l1, l2]</param>
        /// <param name="minibatchSize">Minipack size</param>
        /// <param name="inputDim">Dimension of signs (capacity)</param> 
        /// <returns></returns>
        public IEnumerable<Minibatch> ConvertDatasetToMinibatch<T>(IEnumerable<T[]> dataset, int inputDim, int minibatchSize) where T : IConvertible
        {
            var outputDim = (dataset.FirstOrDefault()?.Length ?? 0) - inputDim;
            foreach (var minibatchData in GetSegments(dataset, minibatchSize))
            {
                var features = minibatchData.SelectMany(p => p.Take(inputDim)).ToArray();
                var labels = minibatchData.SelectMany(p => p.Skip(inputDim)).ToArray();

                Minibatch minibatch = new Minibatch();
                minibatch.Size = minibatchData.Count;
                minibatch.Features = Value.CreateBatch(new int[] { inputDim }, features, Device);
                minibatch.Labels = Value.CreateBatch(new int[] { outputDim }, labels, Device);                

                yield return minibatch;
            }
        }
        /// <summary>
        /// Converts a dataset to a set of training examples in 2D for use in CNTK. 
        /// </summary>
        /// <typeparam name="T">Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="features">2D Feature Set</param>
        /// <param name="labels">Set of labels. The dimension of the labels should be the same.</param>
        /// <param name="minibatchSize">Minipack size</param>
        /// <returns></returns>
        public IEnumerable<Minibatch> ConvertDatasetToMinibatch<T>(IEnumerable<T[,]> features, IEnumerable<T[]> labels, int minibatchSize) where T : IConvertible
        {
            var combined = features.Zip(labels, (f, l) => (f, l));
            foreach (var segment in GetSegments(combined, minibatchSize))
            {
                var featuresData = segment.SelectMany(p => MatrixToVector(p.f));
                var labelsData = segment.SelectMany(p => p.l);

                Minibatch minibatch = new Minibatch();
                minibatch.Size = segment.Count;
                minibatch.Features = Value.CreateBatch(new int[] { GetRowsCount(segment[0].f), GetColumnsCount(segment[0].f), 1 }, featuresData, Device);
                minibatch.Labels = Value.CreateBatch(new int[] { segment[0].l.Length }, labelsData, Device);   

                yield return minibatch;
            }
        }
        /// <summary>
        /// Converts a native feature set to a feature set in CNTK format.
        /// </summary>
        /// <typeparam name="T">Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="data">A set of attributes for each example (sample)</param>
        /// <param name="minibatchSize">The size of the packet by which the signs are broken</param>
        /// <returns></returns>
        public IEnumerable<Value> ConvertDataToValue<T>(IEnumerable<T[]> data, int minibatchSize) where T : IConvertible
        {
            int inputDim = data.FirstOrDefault()?.Length ?? 0;
            foreach (var segment in GetSegments(data, minibatchSize))
            {
                var features = segment.SelectMany(p => p).ToArray();
                var value = Value.CreateBatch(new int[] { inputDim }, features, Device);
                yield return value;
            }
        }
        /// <summary>
        /// Converts a native feature set (sequence) into a feature set in CNTK format.
        /// </summary>
        /// <typeparam name="T">Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="data">A set of attributes for each example (sample), where the example is a sequence</param>
        /// <param name="minibatchSize">The size of the packet by which the signs are broken</param>
        /// <returns></returns>
        public IEnumerable<Value> ConvertDataToValue<T>(IEnumerable<IList<T[]>> data, int minibatchSize) where T : IConvertible
        {
            int inputDim = data.FirstOrDefault()[0].Length;
            foreach (var segment in GetSegments(data, minibatchSize))
            {
                //{}-miniBatch, ()-sequence, []-features.  
                //{ ([inputDim], [inputDim], [inputDim]), ([inputDim], [inputDim]) } => { [inputDim * 3], [inputDim * 2] }
                var featuresTransformed = segment.Select(p => p.SelectMany(q => q));
                var value = Value.CreateBatchOfSequences(new[] { inputDim }, featuresTransformed, Device);
                yield return value;
            }
        }
        /// <summary>
        /// Converts a native feature set (2D) to a feature set in CNTK format.
        /// </summary>
        /// <typeparam name="T">Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="data">A set of attributes for each example (sample), where the example is 2D</param>
        /// <param name="minibatchSize">The size of the packet by which the signs are broken</param>
        /// <returns></returns>
        public IEnumerable<Value> ConvertDataToValue<T>(IEnumerable<T[,]> data, int minibatchSize) where T : IConvertible
        {
            foreach (var segment in GetSegments(data, minibatchSize))
            {
                var features = segment.SelectMany(p => MatrixToVector(p));
                var value = Value.CreateBatch(new int[] { GetRowsCount(segment[0]), GetColumnsCount(segment[0]), 1 }, features, Device);
                yield return value;
            }
        }
        /// <summary>
        /// Converts a dataset into sets of training examples for use in CNTK. Used to train models with multiple outputs.
        /// </summary>
        /// <typeparam name="T">Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="features">A set of attributes for each example (sample)</param>
        /// <param name="labels">A set of labels for each output of the model, the dimension for each output may be different.</param>
        /// <param name="minibatchSize">Minipack size</param>
        /// <returns></returns>
        public IEnumerable<MinibatchMultiOutput> ConvertDatasetToMinibatchMultiOutput<T>(IEnumerable<T[]> features, IEnumerable<T[][]> labels, int minibatchSize)
        {
            int inputDim = features.FirstOrDefault()?.Length ?? 0;
            int outputCount = labels.FirstOrDefault()?.Length ?? 0;
            var combined = features.Zip(labels, (f, l) => (f, l));
            foreach (var segment in GetSegments(combined, minibatchSize))
            {
                var featuresData = segment.SelectMany(p => p.f);
                var labelsData = new T[outputCount][];
                for (int i = 0; i < outputCount; i++)
                {
                    labelsData[i] = segment.SelectMany(p => p.l[i]).ToArray();
                }                

                MinibatchMultiOutput minibatch = new MinibatchMultiOutput();
                minibatch.Size = segment.Count;
                minibatch.Features = Value.CreateBatch(new int[] { inputDim }, featuresData, Device);
                minibatch.Labels = labelsData
                    .Select(label => Value.CreateBatch(new int[] { label.Length / segment.Count }, label, Device))
                    .ToArray();

                yield return minibatch;
            }
        }
        /// <summary>
        /// Converts a dataset into sets of training examples for use in CNTK. Used to train recursive models with multiple outputs.
        /// </summary>
        /// <typeparam name="T">Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="features">A set of attributes for each example (sample), with an example - a sequence</param>
        /// <param name="labels">A set of labels for each output of the model, the dimension for each output may be different.</param>
        /// <param name="minibatchSize">Minipack size</param>
        /// <returns></returns>
        public IEnumerable<MinibatchMultiOutput> ConvertDatasetToMinibatchMultiOutput<T>(IEnumerable<IList<T[]>> features, IEnumerable<T[][]> labels, int minibatchSize)
        {
            int inputDim = features.FirstOrDefault()?[0].Length ?? 0;
            int outputCount = labels.FirstOrDefault()?.Length ?? 0;
            var combined = features.Zip(labels, (f, l) => (f, l));
            foreach (var segment in GetSegments(combined, minibatchSize))
            {
                var featuresData = segment.Select(p => p.f.SelectMany(q => q));
                var labelsData = new T[outputCount][];
                for (int i = 0; i < outputCount; i++)
                {
                    labelsData[i] = segment.SelectMany(p => p.l[i]).ToArray();
                }

                MinibatchMultiOutput minibatch = new MinibatchMultiOutput();
                minibatch.Size = segment.Count;
                minibatch.Features = Value.CreateBatch(new int[] { inputDim }, featuresData, Device);
                minibatch.Labels = labelsData
                    .Select(label => Value.CreateBatch(new int[] { label.Length / segment.Count }, label, Device))
                    .ToArray();

                yield return minibatch;
            }
        }
        /// <summary>
        /// Converts a 2D dataset into training case sets for use in CNTK. Used to train models with multiple outputs.
        /// </summary>
        /// <typeparam name="T">Supported by<seealso cref="float"/>, <seealso cref="double"/></typeparam>
        /// <param name="features">A set of attributes for each example (sample) in 2D</param>
        /// <param name="labels">A set of labels for each output of the model, the dimension for each output may be different.</param>
        /// <param name="minibatchSize">Minipack size</param>
        /// <returns></returns>
        public IEnumerable<MinibatchMultiOutput> ConvertDatasetToMinibatchMultiOutput<T>(IEnumerable<T[,]> features, IEnumerable<T[][]> labels, int minibatchSize) where T:IConvertible
        {            
            int outputCount = labels.FirstOrDefault()?.Length ?? 0;
            var combined = features.Zip(labels, (f, l) => (f, l));
            foreach (var segment in GetSegments(combined, minibatchSize))
            {
                var featuresData = segment.SelectMany(p => MatrixToVector(p.f));
                var labelsData = new T[outputCount][];
                for (int i = 0; i < outputCount; i++)
                {
                    labelsData[i] = segment.SelectMany(p => p.l[i]).ToArray();
                }

                MinibatchMultiOutput minibatch = new MinibatchMultiOutput();
                minibatch.Size = segment.Count;
                minibatch.Features = Value.CreateBatch(new int[] { GetRowsCount(segment[0].f), GetColumnsCount(segment[0].f), 1 }, featuresData, Device);
                minibatch.Labels = labelsData
                    .Select(label => Value.CreateBatch(new int[] { label.Length / segment.Count }, label, Device))
                    .ToArray();

                yield return minibatch;
            }
        }


        #region IDisposable Support
        private bool disposedValue = false; // To identify redundant calls

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    // TODO: release managed state (managed objects).
                    Device.Dispose();
                }

                // TODO: Release unmanaged resources (unmanaged objects) and override the completion method below.
                // TODO: set large fields to NULL.
                Device = null;
                disposedValue = true;
            }
        }

        // TODO: Override the completion method only if Dispose (bool disposing) above includes code to free unmanaged resources.
        ~DataConverter()
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

    /// <summary>
    /// Represents a stack of data for training
    /// </summary>
    public class Minibatch
    {
        /// <summary>
        /// Pack size (number of training examples per pack)
        /// </summary>
        public int Size { get; set; }
        /// <summary>
        /// Signs
        /// </summary>
        public Value Features { get; set; }
        /// <summary>
        /// Class Labels / Continuous Label Values
        /// </summary>
        public Value Labels { get; set; }
    }
    /// <summary>
    /// Represents a stack of data for training models with multiple outputs
    /// </summary>
    public class MinibatchMultiOutput
    {
        /// <summary>
        /// Pack size (number of training examples per pack)
        /// </summary>
        public int Size { get; set; }
        /// <summary>
        /// Signs
        /// </summary>
        public Value Features { get; set; }
        /// <summary>
        /// Class labels / continuous label values, for each model output
        /// </summary>
        public Value[] Labels { get; set; }
    }
}
