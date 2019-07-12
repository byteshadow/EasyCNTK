using CNTK;

namespace EasyCNTK.Layers
{
    public sealed class Embedding : Layer
    {
        private readonly int _dimension;
        public Embedding(int dimension)
        {
            _dimension = dimension;
        }

        public static Function Build(Variable input, int embeddingDim, DeviceDescriptor device)
        {
            System.Diagnostics.Debug.Assert(input.Shape.Rank == 1);

            var inputDim = input.Shape[0];
            var embeddingParameters = new Parameter(new[] { embeddingDim, inputDim }, input.DataType, CNTKLib.GlorotUniformInitializer(), device);
            return CNTKLib.Times(embeddingParameters, input);
        }

        public override Function Create(Function input, DeviceDescriptor device)
        {
            return Build(input, _dimension, device);
        }

        public override string GetDescription()
        {
            return "Embedding";
        }
    }
}