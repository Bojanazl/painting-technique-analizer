using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class Neuron
    {
        private IActivationFunction _activationFunction;
        private IInputFunction _inputFunction;

        public List<ISynapse> Inputs { get; set; }
        public List<ISynapse> Outputs { get; set; }

        public Guid Id { get; private set; }

        public double PreviousPartialDerivate { get; set; }

        public Neuron(IActivationFunction activationFunction, IInputFunction inputFunction)
        {
            Id = Guid.NewGuid();
            Inputs = new List<ISynapse>();
            Outputs = new List<ISynapse>();

            _activationFunction = activationFunction;
            _inputFunction = inputFunction;
        }

        //For neurons of input layer
        public void AddInputNeuron(Neuron inputNeuron)
        {
            var synapse = new Synapse(inputNeuron, this);
            Inputs.Add(synapse);
            inputNeuron.Outputs.Add(synapse);
        }

        public void AddOutputNeuron(Neuron outputNeuron)
        {
            var synapse = new Synapse(this, outputNeuron);
            Outputs.Add(synapse);
            outputNeuron.Inputs.Add(synapse);
        }

        public double CalculateNetInput(List<ISynapse> inputs)
        {
            return inputs.Select(x => x.Weight * x.GetOutput()).Sum();
        }

        public double CalculateOutput()
        {
            return _activationFunction.CalculateOutput(_inputFunction.CalculateInput(this.Inputs));
        }

        public void AddInputSynapse(double inputValue)
        {
            var inputSynapse = new InputSynapse(this, inputValue);
            Inputs.Add(inputSynapse);
        }

        public void PushValueOnInput(double inputValue)
        {
            ((InputSynapse)Inputs.First()).Output = inputValue;
        }
    }
}
