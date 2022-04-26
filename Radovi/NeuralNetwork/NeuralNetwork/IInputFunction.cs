using System.Collections.Generic;

namespace NeuralNetwork
{
    public interface IInputFunction
    {
        double CalculateInput(List<ISynapse> inputs);
    }
}