#include <Eigen/Dense>
class NeuralNetwork {
public:
    Eigen::MatrixXd net;
    NeuralNetwork(unsigned inputDim, unsigned outputDim) {
        net = Eigen::MatrixXd::Zero(inputDim, outputDim);
    };

    void compute(Eigen::Map<const Eigen::VectorXd> input, 
            Eigen::Map<Eigen::VectorXd>  output) {
        output = net * input;
    }
};