#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include <fstream>
#include <algorithm>
#include <random>
#include <sstream>
#include <iterator>
#include <stdexcept>


std::vector<float> pointwiseMult(std::vector<float> vecA, std::vector<float> vecB);
std::vector<float> getUnitVectorIfMagGreaterThan(std::vector<float> input, int maxMagnitude);

enum activationFunction {
    relu,
    sigmoid,
    step,
    nothing,
    softmax
};
struct DataPoint {
    std::vector<float> features;
    int label;
    friend std::ostream& operator<<(std::ostream& out, DataPoint &dp);
};


class Vector {
    public:
    int length;
    std::vector<float> weights;
    Vector(int l, std::vector<float> w) : length(l), weights(w){}
    Vector(int l): length(l) {
        for (int i = 0; i < length; i++) {
            weights.push_back(0);
        }
    }
    float dot(Vector a);
    friend std::ostream& operator<<(std::ostream& out, Vector& v);
};
class Matrix {
    public:
    int nCol;
    int nRow;
    std::vector<float> weights;
    Matrix();
    Matrix(int r, int c);
    Matrix(int r, int c, std::vector<float> w);
    Matrix(std::vector<float> vec1, std::vector<float> vec2);
    Matrix(const Matrix &m);
    Matrix copy();
    Vector getRow(int i);
    Vector getCol(int i);

    void setValAt(int r, int c, float val);
    float getValAt(int r, int c) const;

    Matrix transpose();
    void add(Matrix& m);
    friend std::ostream& operator<<(std::ostream& out, Matrix &m);
    friend Matrix operator*(Matrix& A, Matrix& B);
};

std::vector<float> getFeatureWiseAverage(Matrix m);

Matrix getDeltaCrossEntropy(Matrix softmaxOutputs, std::vector<int> labels);

class DataSet {
    public:
    std::vector<DataPoint> points;
    std::vector<int> classes; // contains all the possible classes.
    std::vector<DataPoint> trainData;
    std::vector<int> trainLabels;

    std::vector<DataPoint> testData;
    std::vector<int> testLabels;
    float trainTestSplit = 0.9;
    DataSet(int numPoints, int numC);
    DataSet(std::vector<DataPoint> p); 
    void shuffleData ();
    void initializeTrainTestData();

    std::vector<int> getLabels();

    
};

Matrix getMatrix(std::vector<DataPoint> pointSubset);

std::vector<float> getSoftmax(std::vector<float> x);

class Layer {
    public:
    std::vector<float> biases;
    Matrix weightMatrix;
    Matrix previousOutput;
    activationFunction activationFunctionId;

    // values used in backpropagation
    Matrix deltaWeights;
    std::vector<float> deltaBiases;
    std::vector<float> nodeLocalGradients; // Note, also serves as a delta value for biases.

    Layer(int numInputFeatures, int numOutputFeatures);
    Layer(Matrix m, std::vector<float> b, activationFunction fId);


    std::vector<float> activate(std::vector<float> row);
    
    Matrix forward(Matrix &input);

    std::vector<float> backward(Layer& leftLayer, int sampleIdx);
    void setLocalGradients(std::vector<float> localGradients);
    void updateWeightsAndBiases(int numSamples);

    friend std::ostream& operator<<(std::ostream& out, Layer& l);
}; // class Layer

class Network {
    public:
    std::vector<Layer> layers;
    Matrix forward(Matrix input);

    /**
     * @brief set deltaWeights and deltaBiases for each layer to be full of zeroes.
     * 
     */
    void initailizeDeltaWeightsAndBiases();

    /**
     * @brief Run backpropagation for a batch
     * 
     * @param labels 
     */
    void backpropagation(Matrix batchInput, std::vector<int> labels);
    int doInference(std::vector<float> input);
    void saveToFile(std::string filename);

}; // class Network
Network createNetworkFromFile(std::string filename);
float clipVal(float val);
float getBatchLoss(Matrix output, std::vector<int> labels);
float meanLoss(std::vector<float> losses);
std::vector<int> getPredictions(Matrix output);
float accuracy(Matrix output, std::vector<int> labels);

class Trainer {
    public:
    Network network;
    DataSet dataset;
    Matrix xVal;
    std::vector<int> yVal;
    Trainer(Network& n, DataSet d );
    void train(int batchSize, int nEpochs);
};

void testVerySimpleData();

void testSpiralData();
/**
 * @brief Simple average pooling for square inputs for compression the number of features for learning MNIST.
 * returns a compressed version of the input by a factor of compressionFactor.
 * @param inputFeatures 
 * @param compressionFactor 
 * @return std::vector<float>
 */
std::vector<float> pool(std::vector<float> inputFeatures, int compressionFactor);
void testMnistData();
void testReadingNetworkFromFile(std::string filename);

