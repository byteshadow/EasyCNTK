//
// Copyright (c) Stanislav Grigoriev. All rights reserved.
// grigorievstas9@gmail.com 
// https://github.com/StanislavGrigoriev/EasyCNTK
//
// Copyright (c) Microsoft. All rights reserved.
//
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//
using System;
using System.Collections.Generic;
using CNTK;

namespace EasyCNTK.Layers
{
    /// <summary>
    /// Implements the LSTM layer.
    /// The cellular state (C) has an independent dimension, all gates have an output dimension (H), scaling is performed directly when writing to the cellular state.
    /// The input (X [t]) is scaled to the output (H [t-1]) and summed (X [t] + H [t-1]), the memory cell (C) is scaled to the output (H). 
    /// </summary>
    public sealed class LSTMv1 : Layer
    {
        private int _lstmOutputDim;
        private int _lstmCellDim;        
        private bool _useShortcutConnections;
        private bool _isLastLstmLayer;
        private string _name;
        private Layer _selfStabilizerLayer;

        /// <summary>
        /// Creates a LSTM cell that implements one repetition step in a recursive network.
        /// It takes the previous state of the cell (c - cell state) and output (h - hidden state) as arguments.
        /// Returns a tuple of the new state of the cell (c - cell state) and exit (h - hidden state).      
        /// </summary>
        /// <param name="input">Entrance to LSTM (X at step t)</param>
        /// <param name="prevOutput">The previous state of the output LSTM (h at step t-1)</param>
        /// <param name="prevCellState">The previous state of the LSTM cell (s in step t-1)</param>
        /// <param name="useShortcutConnections">Specifies whether to create ShortcutConnections for this cell.</param>
        /// <param name="selfStabilizerLayer">Self-stabilization layer to the prevOutput and prevCellState inputs</param>
        /// <param name="device">Device for calculations</param>
        /// <returns></returns>
        private static Tuple<Function, Function> LSTMCell(Variable input, Variable prevOutput,
            Variable prevCellState,  bool useShortcutConnections, Layer selfStabilizerLayer, DeviceDescriptor device)
        {
            int lstmOutputDimension = prevOutput.Shape[0];
            int lstmCellDimension = prevCellState.Shape[0];
            bool hasDifferentOutputAndCellDimension = lstmCellDimension != lstmOutputDimension;

            DataType dataType = input.DataType;

            if (selfStabilizerLayer != null)
            {
                prevOutput = selfStabilizerLayer.Create(prevOutput, device);
                prevCellState = selfStabilizerLayer.Create(prevCellState, device);
            }

            uint seed = CNTKLib.GetRandomSeed();
            //create an input projection of data from the input X [t] and the hidden state H [t-1]
            Func<int, Variable> createInput = (outputDim) =>
            {
                var inputWeigths = new Parameter(new[] { outputDim, NDShape.InferredDimension }, dataType, CNTKLib.GlorotUniformInitializer(1.0, 1, 0, seed++), device);
                var inputBias = new Parameter(new[] { outputDim }, dataType, CNTKLib.GlorotUniformInitializer(1.0, 1, 0, seed++), device);
                var inputToCell = CNTKLib.Times(inputWeigths, input) + inputBias;

                var gateInput = CNTKLib.Plus(inputToCell, prevOutput);
                return gateInput;
            };

            Func<int, Variable, Variable> createProjection = (targetDim, variableNeedsToProjection) =>
            {
                var cellWeigths = new Parameter(new[] { targetDim, NDShape.InferredDimension }, dataType, CNTKLib.GlorotUniformInitializer(1.0, 1, 0, seed++), device);
                var projection = CNTKLib.Times(cellWeigths, variableNeedsToProjection);
                return projection;
            };

            Variable forgetProjection = createInput(lstmOutputDimension);
            Variable inputProjection = createInput(lstmOutputDimension);
            Variable candidateProjection = createInput(lstmOutputDimension);
            Variable outputProjection = createInput(lstmOutputDimension);

            Function forgetGate = CNTKLib.Sigmoid(forgetProjection); // forget valve (from the input in step t)  
            Function inputGate = CNTKLib.Sigmoid(inputProjection); //input gate (from the input in step t)         
            Function candidateGate = CNTKLib.Tanh(candidateProjection); //the candidate selection gate for storing in the cellular state (from the input in step t)
            Function outputGate = CNTKLib.Sigmoid(outputProjection); //output gate (from the input in step t)  

            forgetGate = hasDifferentOutputAndCellDimension ? createProjection(lstmCellDimension, forgetGate) : (Variable)forgetGate;
            Function forgetState = CNTKLib.ElementTimes(prevCellState, forgetGate); //forget what you need to forget in the cellular state

            Function inputState = CNTKLib.ElementTimes(inputGate, candidateProjection); //we get what we need to save in the cellular state (from the input in step t) 
            inputState = hasDifferentOutputAndCellDimension ? createProjection(lstmCellDimension, inputState) : (Variable)inputState;
            Function cellState = CNTKLib.Plus(forgetState, inputState); //add new information to the cellular state

            Variable cellToOutputProjection = hasDifferentOutputAndCellDimension ? createProjection(lstmOutputDimension, cellState) : (Variable)cellState;
            Function h = CNTKLib.ElementTimes(outputGate, CNTKLib.Tanh(cellToOutputProjection)); //get exit / hidden state
            Function c = cellState;

            if (useShortcutConnections)
            {
                var forwarding = input;
                var inputDim = input.Shape[0];
                if (inputDim != lstmOutputDimension)
                {
                    var scales = new Parameter(new[] { lstmOutputDimension, inputDim }, dataType, CNTKLib.UniformInitializer(seed++), device);
                    forwarding = CNTKLib.Times(scales, input);
                }
                h = CNTKLib.Plus(h, forwarding);
            }

            return new Tuple<Function, Function>(h, c);
        }
        private static Tuple<Function, Function> LSTMComponent(Variable input, NDShape outputShape,
            NDShape cellShape, Func<Variable, Function> recurrenceHookH, Func<Variable, Function> recurrenceHookC,
            bool useShortcutConnections, Layer selfStabilizerLayer, DeviceDescriptor device)
        {
            var dh = Variable.PlaceholderVariable(outputShape, input.DynamicAxes);
            var dc = Variable.PlaceholderVariable(cellShape, input.DynamicAxes);

            var lstmCell = LSTMCell(input, dh, dc, useShortcutConnections, selfStabilizerLayer, device);
            var actualDh = recurrenceHookH(lstmCell.Item1);
            var actualDc = recurrenceHookC(lstmCell.Item2);

            (lstmCell.Item1).ReplacePlaceholders(new Dictionary<Variable, Variable> { { dh, actualDh }, { dc, actualDc } });
            return new Tuple<Function, Function>(lstmCell.Item1, lstmCell.Item2);
        }
        /// <summary>
        /// Creates an LSTM layer. 
        /// The cellular state (C) has an independent dimension, all gates have an output dimension (H), scaling is performed directly when writing to the cellular state.
        /// The input (X [t]) is scaled to the output (H [t-1]) and summed (X [t] + H [t-1]), the memory cell (C) is scaled to the output (H). 
        /// </summary>
        /// <param name="input">Entrance (X)</param>
        /// <param name="lstmDimension">Output layer depth (H)</param>        
        /// <param name="cellDimension">Bit depth of the inner layer of the memory cell, if 0 - sets the bit depth of the output layer (C)</param>
        /// <param name="useShortcutConnections">If true, use input forwarding parallel to the layer. Enabled by default.</param>
        /// <param name="selfStabilizerLayer">Self-stabilization layer</param>
        /// <param name="isLastLstm">Indicates whether this will be the last of the LSTM layers (the next layers on the network are non-recursive). In order to join LSTM layers one after another, all layers except the last one must be set to false</param>
        /// <param name="outputName">layer name</param>
        /// <returns></returns>
        public static Function Build(Function input, int lstmDimension, DeviceDescriptor device, int cellDimension = 0, bool useShortcutConnections = true, bool isLastLstm = true, Layer selfStabilizerLayer = null, string outputName = "")
        {
            if (cellDimension == 0) cellDimension = lstmDimension;
            
            Func<Variable, Function> pastValueRecurrenceHook = (x) => CNTKLib.PastValue(x);

            var lstm = LSTMComponent(input, new int[] { lstmDimension }, new int[] { cellDimension },
                    pastValueRecurrenceHook, pastValueRecurrenceHook, useShortcutConnections, selfStabilizerLayer, device)
                .Item1;

            lstm = isLastLstm ? CNTKLib.SequenceLast(lstm) : lstm;
            return Function.Alias(lstm, outputName);
        }

        public override Function Create(Function input, DeviceDescriptor device)
        {
            return Build(input, _lstmOutputDim, device, _lstmCellDim, _useShortcutConnections, _isLastLstmLayer, _selfStabilizerLayer, _name);
        }
        /// <summary>
        /// Creates an LSTM layer. 
        /// The cellular state (C) has an independent dimension, all gates have an output dimension (H), scaling is performed directly when writing to the cellular state.
        /// The input (X [t]) is scaled to the output (H [t-1]), the memory cell (C) is scaled to the output (H). 
        /// </summary>
        /// <param name="lstmOutputDim">Output layer depth (H)</param>        
        /// <param name="lstmCellDim">Bit depth of the inner layer of the memory cell, if 0 - sets the bit depth of the output layer (C)</param>
        /// <param name="useShortcutConnections">If true, use input forwarding parallel to the layer. Enabled by default.</param>
        /// <param name="selfStabilizerLayer">Self-stabilization layer</param>
        /// <param name="isLastLstm">Indicates whether this will be the last of the LSTM layers (the next layers on the network are non-recursive). In order to join LSTM layers one after another, all layers except the last one must be set to false</param>
        /// <param name="name"></param>
        public LSTMv1(int lstmOutputDim, int lstmCellDim = 0, bool useShortcutConnections = true, bool isLastLstm = true, Layer selfStabilizerLayer = null, string name = "LSTMv1")
        {
            _lstmOutputDim = lstmOutputDim;
            _lstmCellDim = lstmCellDim == 0 ? _lstmOutputDim : _lstmCellDim; ;
            _useShortcutConnections = useShortcutConnections;
            _isLastLstmLayer = isLastLstm;
            _selfStabilizerLayer = selfStabilizerLayer;
            _name = name;
        }

        public override string GetDescription()
        {
            return $"LSTMv1(C={_lstmCellDim}H={_lstmOutputDim}SC={_useShortcutConnections}SS={_selfStabilizerLayer?.GetDescription() ?? "none"})";
        }
    }
}
