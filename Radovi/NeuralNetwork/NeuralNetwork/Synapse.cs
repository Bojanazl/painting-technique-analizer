using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public interface ISynapse
    {
        double Weight { get; set; }
        double PreviousWeight { get; set; }
        double GetOutput();

        bool IsFromNeuron(Guid fromNeuronId);
        void UpdateWeight(double learningRate, double delta);
    }

    public class Synapse : ISynapse
    {
        internal Neuron _fromNeuron;
        internal Neuron _toNeuron;

        internal static Random rand = new Random();

        public double Weight { get; set; }
        public double PreviousWeight { get; set; }

        public Synapse(Neuron fromNeuron, Neuron toNeuron, double weight)
        {
            _fromNeuron = fromNeuron;
            _toNeuron = toNeuron;

            Weight = weight;
        }

        public Synapse(Neuron fromNeuron, Neuron toNeuron)
        {
            _fromNeuron = fromNeuron;
            _toNeuron = toNeuron;

            Weight = rand.NextDouble() / 2.5 -0.2;
        }

        public double GetOutput()
        {
            return _fromNeuron.CalculateOutput();
        }

        public bool IsFromNeuron(Guid fromNeuronId)
        {
            return _fromNeuron.Id.Equals(fromNeuronId);
        }

        public void UpdateWeight(double learningRate, double delta)
        {
            PreviousWeight = Weight;
            Weight += learningRate * delta;
        }
    }

    public class InputSynapse : ISynapse
    {
        internal Neuron _toNeuron;

        public double Weight { get; set; }
        public double PreviousWeight { get; set; }
        public double Output { get; set; }

        public InputSynapse(Neuron toNeuron)
        {
            _toNeuron = toNeuron;
            Weight = 1;
        }

        public InputSynapse(Neuron toNeuron, double output)
        {
            _toNeuron = toNeuron;
            Output = output;
            Weight = 1;
            PreviousWeight = 1;
        }

        public double GetOutput()
        {
            return Output;
        }

        public bool IsFromNeuron(Guid fromNeuronId)
        {
            return false;
        }

        public void UpdateWeight(double learningRate, double delta)
        {
            throw new InvalidOperationException("Weights are not modified on input connections (always 1)");
        }
    }
}
