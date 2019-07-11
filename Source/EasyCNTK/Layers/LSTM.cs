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
    /// The cellular state (C) has a common dimension - all gates have the dimension of the cellular state, scaling is done only at the entrance and exit of the cell.
    /// The input (X [t] + H [t-1]) is scaled to the memory cell (C [t]), the memory cell is scaled to the output (H [t])
    /// </summary>
    public sealed class LSTM : Layer
    {
        private readonly int _lstmOutputDim;
        private readonly int _lstmCellDim;
        private readonly bool _useShortcutConnections;
        private readonly bool _isLastLstmLayer;
        private readonly string _name;
        private readonly Layer _selfStabilizerLayer;

        /// <summary>
        /// Creates an LSTM cell that implements a single repetition step in a recurrent network.
        /// Takes as arguments the previous states of the cell (c - cell state) and the output (h - hidden state).
        /// Returns the tuple of the new state of the cell (c - cell state) and output (h - hidden state).     
        /// </summary>
        /// <param name = "input"> Input to LSTM (X in step t) </param>
        /// <param name = "prevOutput"> Previous output state of LSTM (h in step t-1) </param>
        /// <param name = "prevCellState"> The previous state of the LSTM cell (as in step t-1) </param>
        /// <param name = "useShortcutConnections"> Specifies whether to create a ShortcutConnections for this cell </param>
        /// <param name = "selfStabilizerLayer"> A layer that implements self-stabilization. If not null, self-stabilization will be applied to the prevOutput and prevCellState inputs </param>
        /// <param name = "device"> Device for calculations </param>
        /// <returns> Function (prev_h, prev_c, input) -> (h, c) which implements one step of repeating LSTM layer </returns>
        private static Tuple<Function, Function> Cell(Variable input, Variable prevOutput, Variable prevCellState, bool useShortcutConnections, Layer selfStabilizerLayer, DeviceDescriptor device)
        {
            int lstmOutputDimension = prevOutput.Shape[0];
            int lstmCellDimension = prevCellState.Shape[0];

            DataType dataType = input.DataType;

            if (selfStabilizerLayer != null)
            {
                prevOutput = selfStabilizerLayer.Create(prevOutput, device);
                prevCellState = selfStabilizerLayer.Create(prevCellState, device);
            }

            uint seed = CNTKLib.GetRandomSeed();
            // create an input data projection for the cell from the input X [t] and the hidden state H [t-1]
            Variable CreateInput(int cellDim, int hiddenDim)
            {
                var inputWeights = new Parameter(new[] { cellDim, NDShape.InferredDimension }, dataType, CNTKLib.GlorotUniformInitializer(1.0, 1, 0, seed++), device);
                var inputBias = new Parameter(new[] { cellDim }, dataType, CNTKLib.GlorotUniformInitializer(1.0, 1, 0, seed++), device);
                var inputToCell = CNTKLib.Times(inputWeights, input) + inputBias;

                var hiddenWeights = new Parameter(new[] { cellDim, hiddenDim }, dataType, CNTKLib.GlorotUniformInitializer(1.0, 1, 0, seed++), device);
                var hiddenState = CNTKLib.Times(hiddenWeights, prevOutput);

                var gateInput = CNTKLib.Plus(inputToCell, hiddenState);
                return gateInput;
            }

            Variable forgetProjection    = CreateInput(lstmCellDimension, lstmOutputDimension);
            Variable inputProjection     = CreateInput(lstmCellDimension, lstmOutputDimension);
            Variable candidateProjection = CreateInput(lstmCellDimension, lstmOutputDimension);
            Variable outputProjection    = CreateInput(lstmCellDimension, lstmOutputDimension);

            Function forgetGate    = CNTKLib.Sigmoid(forgetProjection);// gate "forgetting" (from the input in step t)
            Function inputGate     = CNTKLib.Sigmoid(inputProjection); // input gate (from the input in step t)         
            Function candidateGate = CNTKLib.Tanh(candidateProjection); // valve for selecting candidates for memorization in the cellular state (from the input data in step t)
            Function outputGate    = CNTKLib.Sigmoid(outputProjection); // output gate (from the input in step t)

            Function forgetState = CNTKLib.ElementTimes(prevCellState, forgetGate); // forget what you need to forget in the cellular state
            Function inputState  = CNTKLib.ElementTimes(inputGate, candidateProjection); // we get what we need to save in the cellular state (from the input data in step t) 
            Function cellState   = CNTKLib.Plus(forgetState, inputState); // add new information to the cellular state

            Function h = CNTKLib.ElementTimes(outputGate, CNTKLib.Tanh(cellState)); // get output / hidden state
            Function c = cellState;
            if (lstmOutputDimension != lstmCellDimension)
            {
                Parameter scale = new Parameter(new[] { lstmOutputDimension, lstmCellDimension }, dataType, CNTKLib.GlorotUniformInitializer(1.0, 1, 0, seed++), device);
                h = CNTKLib.Times(scale, h);
            }

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
        private static Tuple<Function, Function> Component(Variable input, NDShape outputShape,
            NDShape cellShape, Func<Variable, Function> recurrenceHookH, Func<Variable, Function> recurrenceHookC,
            bool useShortcutConnections, Layer selfStabilizerLayer, DeviceDescriptor device)
        {
            var dh = Variable.PlaceholderVariable(outputShape, input.DynamicAxes);
            var dc = Variable.PlaceholderVariable(cellShape, input.DynamicAxes);

            var lstmCell = Cell(input, dh, dc, useShortcutConnections, selfStabilizerLayer, device);
            var actualDh = recurrenceHookH(lstmCell.Item1);
            var actualDc = recurrenceHookC(lstmCell.Item2);

            (lstmCell.Item1).ReplacePlaceholders(new Dictionary<Variable, Variable> { { dh, actualDh }, { dc, actualDc } });
            return new Tuple<Function, Function>(lstmCell.Item1, lstmCell.Item2);
        }
        /// <summary>
        /// Creates an LSTM layer.
        /// The cellular state (C) has a common dimension - all gates have the dimension of the cellular state, scaling is done only at the entrance and exit of the cell.
        /// The input (X [t] + H [t-1]) is scaled to the memory cell (C [t]), the memory cell is scaled to the output (H [t])
        /// </summary>
        /// <param name = "input"> Input (X) </param>
        /// <param name = "lstmDimension"> Output layer width (H) </param>
        /// <param name = "cellDimension"> The width of the inner layer of the memory cell, if 0 - set the width of the output layer (C) </param>
        /// <param name = "useShortcutConnections"> If true, use input forwarding parallel to the layer. Enabled by default. </Param>
        /// <param name = "selfStabilizerLayer"> A layer that implements self-stabilization. If not null, self-stabilization will be applied to the prevOutput and prevCellState inputs </param>
        /// <param name = "isLastLstm"> Indicates whether this is the last of the LSTM layers (the next layers in the network are non-recurrent). In order to join LSTM layers one after another, all layers except the last need to be set to false </param>
        /// <param name = "outputName"> layer name </param>
        /// <returns></returns>
        public static Function Build(Function input, int lstmDimension, DeviceDescriptor device, int cellDimension = 0, bool useShortcutConnections = true, bool isLastLstm = true, Layer selfStabilizerLayer = null, string outputName = "")
        {
            if (cellDimension == 0) cellDimension = lstmDimension;
            Func<Variable, Function> pastValueRecurrenceHook = (x) => CNTKLib.PastValue(x);

            var lstm = Component(input, new int[] { lstmDimension }, new int[] { cellDimension },
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
        /// The cellular state (C) has a common dimension - all gates have the dimension of the cellular state, scaling is done only at the entrance and exit of the cell.
        /// The input (X [t] + H [t-1]) is scaled to the memory cell (C [t]), the memory cell is scaled to the output (H [t])
        /// </summary>
        /// <param name = "lstmOutputDim"> Output layer width (H) </param>
        /// <param name = "lstmCellDim"> The width of the inner layer of the memory cell, if 0 - sets the width of the output layer (C) </param>
        /// <param name = "useShortcutConnections"> If true, use input forwarding parallel to the layer. Enabled by default. </Param>
        /// <param name = "selfStabilizerLayer"> A layer that implements self-stabilization. If not null, self-stabilization will be applied to the C [t-1] and H [t-1] inputs </param>
        /// <param name = "isLastLstm"> Indicates whether this is the last of the LSTM layers (the next layers in the network are non-recurrent). In order to join LSTM layers one after another, all layers except the last need to be set to false </param>
        /// <param name = "name"> </param>
        public LSTM(int lstmOutputDim, int lstmCellDim = 0, bool useShortcutConnections = true, bool isLastLstm = true, Layer selfStabilizerLayer = null, string name = "LSTM")
        {
            _lstmOutputDim = lstmOutputDim;
            _lstmCellDim = lstmCellDim == 0 ? _lstmOutputDim : _lstmCellDim;
            _useShortcutConnections = useShortcutConnections;
            _isLastLstmLayer = isLastLstm;
            _selfStabilizerLayer = selfStabilizerLayer;
            _name = name;
        }

        public override string GetDescription()
        {
            return $"LSTM(C={_lstmCellDim}H={_lstmOutputDim}SC={_useShortcutConnections}SS={_selfStabilizerLayer?.GetDescription() ?? "none"})";
        }
    }
}
