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
#include "network.hpp"


// random device class instance, source of 'true' randomness for initializing random seed
std::random_device rd; 

// Mersenne twister PRNG, initialized with seed from previous random device instance
std::mt19937 gen(rd());
std::normal_distribution<float> d(0, 1.0f);

std::vector<float> pointwiseMult(std::vector<float> vecA, std::vector<float> vecB) {
    /**
     * @brief Multiply two vectors pointwise and recieve a new vector of the same size.
     * 
     */
    std::vector<float> result;
    if(vecA.size() != vecB.size()) {
        throw std::invalid_argument("pointwiseMult: pointwise multiplication error. Both vectors must be of equal size.");
    }
    for(int i = 0; i < vecA.size(); i++) {
        result.push_back(vecA[i] * vecB[i]);
    }
    return result;
}

std::vector<float> getUnitVectorIfMagGreaterThan(std::vector<float> input, int maxMagnitude) {
    std::vector<float> result(input.size());
    float magnitude = 0;
    for(int i = 0; i < input.size(); i++) {
        magnitude += (input[i] * input[i]);
    }
    magnitude = std::sqrt(magnitude);
    if(magnitude <= maxMagnitude) return input;
    for(int i = 0; i < input.size(); i++) {
        result[i] = input[i]/magnitude;
    }
    return result;
}

// Matrix pointwiseMult(Matrix a, Matrix b) {
//     if(a.nRow != b.nRow && a.nCol != b.nCol) {
//         throw "pontwiseMullt: pointwise multiplication error. Both matrices must have equal dimensions.";
//     }
//     Matrix result(a);
//     for(int i = 0; i < result.weights.size(); i++) {
//         result.weights[i] *= b.weights[i];
//     }
//     return result;
// }

const float LEARNING_RATE = 0.1f;


float Vector::dot(Vector a) {
    if(length != a.length) {
        throw std::invalid_argument("lengths do not match, dot product could not be calculated");
    }
    float res = 0;
    for(int i = 0; i < length; i++) {
        res += this->weights[i] * a.weights[i];
    }
    return res;
}


Matrix::Matrix(){
    nCol = 0;
    nRow = 0;
}
Matrix::Matrix(int r, int c) : nCol(c), nRow(r) {
    for(int i = 0; i < c*r; i++) {
        weights.push_back(0);
    }
}
Matrix::Matrix(int r, int c, std::vector<float> w) : nCol(c), nRow(r), weights(w) {
}
Matrix::Matrix(std::vector<float> vec1, std::vector<float> vec2) {
    /**
     * @brief Create a matrix from two vectors by multipling them into a grid. Used to calculate delta weight values.
     * 
     */
    nRow = vec1.size();
    nCol = vec2.size();
    for (int i = 0; i < nRow; i++) {
        for (int j = 0; j < nCol; j++) {
            weights.push_back(vec1[i] * vec2[j]);
        }
    }
}
Matrix::Matrix(const Matrix &m) {
    nCol = m.nCol;
    nRow = m.nRow;
    weights = {};
    for(int i=0; i < m.nRow; i++) {
        for(int j = 0; j < m.nCol; j++)  {
            weights.push_back(m.getValAt(i, j));
        }
    }
}
Matrix Matrix::copy() {
    Matrix m;
    m.nCol = nCol;
    m.nRow = nRow;
    m.weights = {};
    for(int i = 0; i < weights.size(); i++) {
        m.weights.push_back(weights[i]);
    }
    return m;
}
Vector Matrix::getRow(int i) {
    if(i > nRow || i < 0) {
        throw std::invalid_argument("getRow: invalid row index");
    }
    std::vector<float> resWeights;
    int initialPos = i*nCol;
    int endPos = initialPos + nCol;
    for(int j = initialPos; j < endPos; j++) {
        resWeights.push_back(weights.at(j));
    }
    Vector v(nCol, resWeights);
    return v;
    
}
Vector Matrix::getCol(int i) {
    if(i > nCol || i < 0) {
        throw std::invalid_argument("getCol: invalid colummn id");
    }
    std::vector<float> resWeights;
    int initialPos = i;
    int finalPos = nCol * nRow + i;
    for(int j = initialPos; j < finalPos; j+=nCol) {
        resWeights.push_back(weights.at(j));
    }
    Vector v(nRow, resWeights);
    return v;
}

void Matrix::setValAt(int r, int c, float val) {
    weights[r*nCol + c] = val;
}
float Matrix::getValAt(int r, int c) const {
    return weights[r*nCol + c];
}


std::vector<float> getFeatureWiseAverage(Matrix m) {
    /**
     * @brief Gets the average per feature value for an input matrix m.
     * 
     */
    std::vector<float> result;
    float inv_size = 1 / m.nRow;
    for(int i = 0; i < m.nCol; i++) {
        result.push_back(0);
    }

    for(int i = 0; i < m.nRow; i++) {
        for(int j = 0; j < m.nCol; j++) {
            result[j] += (m.getValAt(i, j) * inv_size);
        }    
    }
    return result;
}

Matrix getDeltaCrossEntropy(Matrix softmaxOutputs, std::vector<int> labels) {
    /**
     * @brief Calculates the gradient values for the output layer assuming softmax activation.
     * Calculates all the output gradients for a batch.
     * 
     */
    Matrix batchDeltas(softmaxOutputs);

    for(int i = 0; i < labels.size(); i++) {
        batchDeltas.setValAt(labels[i], i, batchDeltas.getValAt(labels[i], i) -1);
    }
    return batchDeltas;
}


DataSet::DataSet(int numPoints, int numC){
    /**
     * @brief Creates a 2d swirled data set with numPoints number of points and numC number of classes.
     * 
     */
    for(int i = 0; i < numC; i++) {
        classes.push_back(i);
    }
    int ix;
    for(int i = 0; i < numC; i++) {
        for(int ix = 0; ix < numPoints; ix++) {
            float r = (float) (ix) / numPoints;
            float t = (float)(((i+1)*4 - (i * 4)) *(i+1)* ix)/numPoints + static_cast <float> (std::rand()) / static_cast <float> (RAND_MAX) * 0.5f;
            DataPoint dp({{(float) (r*sin(t*2.5)), (float)(r*cos(t*2.5))}, i});
            points.push_back(dp);
        }
    }

    initializeTrainTestData();
}
DataSet::DataSet(std::vector<DataPoint> p) : points(p){
    initializeTrainTestData();
} 
void DataSet::shuffleData (){
    /**
     * @brief Shuffle the data points in place
     * cf https://stackoverflow.com/questions/6926433/how-to-shuffle-a-stdvector
     */
    auto rng = std::default_random_engine {};
    std::shuffle(std::begin(points), std::end(points), rng);
}
void DataSet::initializeTrainTestData() {
    shuffleData();
    int splitBoundaryIdx = floor(trainTestSplit * points.size());
    std::vector<DataPoint> trainPoints;
    for(int i = 0; i < splitBoundaryIdx; i++) {
        trainPoints.push_back(points[i]);
    }
    trainData = trainPoints;
    
    std::vector<int> tmpTrainLabels;
    for(int i = 0; i < trainPoints.size(); i++) {
        tmpTrainLabels.push_back(trainPoints[i].label);
    }
    trainLabels = tmpTrainLabels;

    std::vector<DataPoint> testPoints;
    for(int i = splitBoundaryIdx; i < points.size(); i++) {
        testPoints.push_back(points[i]);
    }
    testData = testPoints;

    std::vector<int> tmpTestLabels;
    for(int i = 0; i < testPoints.size(); i++) {
        tmpTestLabels.push_back(testPoints[i].label);
    }
    testLabels = tmpTestLabels;
}

std::vector<int> DataSet::getLabels() {
    std::vector<int> toReturn;
    for(int i = 0; i < points.size(); i++) {
        toReturn.push_back(points[i].label);
    }
    return toReturn;
}


/**
 * @brief Get a matrix of a subset of data. Each sample corresponds to a column vector in the resulting matrix.
 * 
 * @param pointSubset 
 * @return Matrix 
 */
Matrix getMatrix(std::vector<DataPoint> pointSubset) {
    std::vector<float> weights;
    for(int j =0; j < pointSubset[0].features.size(); j++) {
        for(int i = 0; i < pointSubset.size(); i++) {
            weights.push_back(pointSubset[i].features[j]);
        }
    }
    Matrix m(pointSubset[0].features.size(), pointSubset.size(), weights);
    return m;
}


std::vector<float> getSoftmax(std::vector<float> x) {
    std::vector<float> result;
    float totalExp = 0;
    float maxValue = -INFINITY;
    for(int i= 0; i < x.size(); i++) {
        if (x[i] > maxValue) {
            maxValue = x[i];
        }
    }
    for(int i =0; i < x.size(); i++) {
        totalExp += std::exp(x[i] - maxValue);
    }
    float constant = maxValue + std::log(totalExp);
    for(int i = 0; i < x.size(); i++) {
        result.push_back(std::exp(x[i] - constant));
    }
    return result;
}

Layer::Layer(int numInputFeatures, int numOutputFeatures) {
    std::vector<float> initialWeights;
    float inv_sqrt_output = 1.0 / std::sqrt(numOutputFeatures);
    for(int i = 0; i < numInputFeatures * numOutputFeatures; i++) {
        float r = d(gen) * inv_sqrt_output;
        initialWeights.push_back(r);
    }
    std::vector<float> resBiases;
    for(int i =0; i < numOutputFeatures; i++) {
        // float r = static_cast <float> (std::rand()) / static_cast <float> (RAND_MAX);
        resBiases.push_back(0);
    }
    Matrix m(numOutputFeatures, numInputFeatures, initialWeights);
    this->weightMatrix = std::move(m);
    this->biases = std::move(resBiases);
    activationFunctionId = step;
}
Layer::Layer(Matrix m, std::vector<float> b, activationFunction fId): weightMatrix(m), biases(b), activationFunctionId(fId){

}

std::vector<float> Layer::activate(std::vector<float> row) {
    /**
     * @brief Applies an activation function to an output row.
     * 
     */
    float totalExp = 0;
    float rowMax =0;
    if(activationFunctionId == softmax){
        return getSoftmax(row);
    }
    switch(activationFunctionId) {
        case relu:
            for(int i = 0; i < row.size(); i++) {
                row[i] = row[i] > 0 ? row[i] : 0;
            }
            break;
        case step:
            for(int i=0; i < row.size(); i++) {
                row[i] = row[i] > 0 ? 1 : 0;
            }
            break;
        case sigmoid:
            for(int i=0; i < row.size(); i++) {
                if(row[i] < 0) {
                    row[i] = std::exp(row[i])/(1 + std::exp(row[i]));
                }else {
                    row[i] = 1/(1 + std::exp(-row[i]));
                }
            }
            break;
        case softmax:
            for(int i=0; i < row.size(); i++) {
                row[i] = std::exp(row[i]) / totalExp;
            }
            break;

        default: break;
    }
    return row;
}

Matrix Layer::forward(Matrix &input) {
    /**
     * @brief Run input through a forward pass of the layer. Keep track of the avg. output for each node for later backpropagation.
     * 
     */
    // std::cout << "input matrix " << input.nRow <<" " << input.nCol << std::endl;
    // std::cout << "weight matrix " << weightMatrix.nRow <<" " << weightMatrix.nCol << std::endl;
    this->previousOutput = weightMatrix * input;

    for(int sampleIdx = 0; sampleIdx <previousOutput.nCol; sampleIdx++) {
        for(int featureIdx = 0; featureIdx < previousOutput.nRow; featureIdx++) {
            this->previousOutput.setValAt(featureIdx, sampleIdx, this->previousOutput.getValAt(featureIdx, sampleIdx) + this->biases[featureIdx]);
        }
        std::vector<float> activatedCol = activate(this->previousOutput.getCol(sampleIdx).weights);
        for(int featureIdx =0; featureIdx < activatedCol.size(); featureIdx++) {
            this->previousOutput.setValAt(featureIdx, sampleIdx, activatedCol[featureIdx]);
        }
        
    }

    return previousOutput;
}


Matrix Network::forward(Matrix input) {
    layers[0].forward(input);
    // std::cout << "gets here" <<std::endl;
    for(int l = 1; l < layers.size(); l++) {
        layers[l].forward(layers[l-1].previousOutput);
        // std::cout << "gets to " << l << std::endl;
    }
    return layers[layers.size() - 1].previousOutput;
}

/**
 * @brief set deltaWeights and deltaBiases for each layer to be full of zeroes.
 * 
 */
void Network::initailizeDeltaWeightsAndBiases() {
    for(int l = 0; l < layers.size(); l++) {
        Matrix dWeights(layers[l].weightMatrix);
        for(int i =0; i < layers[l].weightMatrix.weights.size(); i++) {
            dWeights.weights[i] = 0;
        }
        layers[l].deltaWeights = std::move(dWeights);
        std::vector<float> dBiases(layers[l].weightMatrix.nRow);
        for(int i = 0; i < dBiases.size(); i++) {
            dBiases[i] = 0;
        }
        layers[l].deltaBiases = std::move(dBiases);
    } 
}

/**
 * @brief Run backpropagation for a batch
 * 
 * @param labels 
 */
void Network::backpropagation(Matrix batchInput, std::vector<int> labels) {
    initailizeDeltaWeightsAndBiases();
    // Get local gradients for output layer for the whole batch.
    Matrix batchCrossEntropyDeltas = getDeltaCrossEntropy(layers[layers.size() - 1].previousOutput, labels);
    
    // Go backward through layers for each sample.
    for(int sampleIdx = 0; sampleIdx < labels.size(); sampleIdx++) {
        // set local gradients for output layer.
        layers[layers.size() -1].setLocalGradients(batchCrossEntropyDeltas.getCol(sampleIdx).weights);
        // handle layers besides input layer
        for(int l = layers.size() - 1; l > 0; l--) {
            // TODO clip these local gradients to mitigate exploding gradient.
            std::vector<float> l_1_local_gradients = layers[l].backward(layers[l-1], sampleIdx);
            layers[l-1].setLocalGradients(l_1_local_gradients);
        }
        // handle input layer
        Matrix inputLayerDeltaWeights(layers[0].nodeLocalGradients, batchInput.getCol(sampleIdx).weights );
        layers[0].deltaWeights.add(inputLayerDeltaWeights);
        for(int i = 0; i < layers[0].deltaBiases.size(); i++) {
            layers[0].deltaBiases[i] += layers[0].nodeLocalGradients[i];
        }
    }

    for(int l =0; l < layers.size(); l++) {
        layers[l].updateWeightsAndBiases(labels.size());
    }
}

int Network::doInference(std::vector<float> input) {
    Matrix inputMat(input.size(), 1, input);
    Matrix resultMat = forward(inputMat);
    std::vector<int> resultLabels = getPredictions(resultMat);
    return resultLabels[0];
}

void Network::saveToFile(std::string filename) {
    std::ofstream networkOutputFile;
    networkOutputFile.open(filename);
    for(Layer layer : layers) {
        networkOutputFile << layer.weightMatrix.nCol << " " << layer.weightMatrix.nRow << "\n";
        for (float weight : layer.weightMatrix.weights) {
            networkOutputFile << weight << " ";
        }
        networkOutputFile << "\n";
        for (float bias : layer.biases) {
            networkOutputFile << bias << " ";
        }
        networkOutputFile << "\n";
        std::string activationFxnName = "sigmoid";
        switch(layer.activationFunctionId) {
            case sigmoid:
                activationFxnName = "sigmoid";
                break;
            case softmax:
                activationFxnName = "softmax";
                break;
        }
        networkOutputFile << activationFxnName << "\n";
    }
    networkOutputFile << "\n";
    networkOutputFile.close();
}

Network createNetworkFromFile(std::string filename) {
    Network network;
    std::ifstream networkInputFile(filename);
    std::string line;
    int counter = 0;
    int nCol = 0;
    int nRow = 0;
    std::vector<float> weights;
    std::vector<float> biases;
    activationFunction activationFunctionType;
    if(networkInputFile.is_open()) {
        while(std::getline(networkInputFile, line)) {
            int lineType = counter % 4;
            std::stringstream ss(line);
            std::istream_iterator<std::string> begin(ss);
            std::istream_iterator<std::string> end;
            std::vector<std::string> featStrings(begin, end);
            switch(lineType) {
                case 0: // weight dimensions 
                    if(nCol > 0) {
                        Matrix weightMatrix(nRow, nCol, weights);
                        Layer l(weightMatrix, biases, activationFunctionType);
                        network.layers.push_back(l);
                        weights.clear();
                        biases.clear();
                        nCol = 0;
                        nRow = 0;
                    }
                    nCol = std::stoi(featStrings[0]);
                    nRow = std::stoi(featStrings[1]);
                    break;
                case 1: // weight values
                    for(int i = 0; i < featStrings.size(); i++) {
                        weights.push_back(std::stof(featStrings[i]));
                    }
                    break;
                case 2: // bias values
                    for(int i = 0; i < featStrings.size(); i++) {
                        biases.push_back(std::stof(featStrings[i]));
                    }
                    break;
                case 3: // activation function type
                    if(featStrings[0] == "sigmoid") {
                        activationFunctionType = sigmoid;
                    } else {
                        activationFunctionType = softmax;
                    }
                    break;
            }
            counter++;
        }
        Matrix weightMatrix(nRow, nCol, weights);
        Layer lastLayer(weightMatrix, biases, activationFunctionType);
        network.layers.push_back(lastLayer);
    }
    networkInputFile.close();
    return network;

}

const float eps = 1e-7;
float clipVal(float val) {
    /**
     * @brief Clips value between 0 and 1 with boundary padding of epsilon.
     * 
     */
    if(val <= 0) return eps;
    if(val >=(1-eps)) return 1-eps;
    return val;
}

float getBatchLoss(Matrix output, std::vector<int> labels) {
    float totalLoss = 0.0f;
    for(int i = 0; i < output.nCol; i++) {
        totalLoss += -std::log(
                clipVal(output.getValAt(labels[i], i)));
    }
    return totalLoss / labels.size();
}

float meanLoss(std::vector<float> losses) {
    float total = 0;
    for(int i = 0; i < losses.size(); i++) {
        total+=losses[i];
    }
    return total/losses.size();
}

std::vector<int> getPredictions(Matrix output) {
    std::vector<int> predictions;
    for(int i = 0; i < output.nCol; i++){
        float curMax = 0;
        int curMaxId = 0;
        float curVal;
        for(int j =0; j < output.nRow; j++) {
            curVal = output.getValAt(j, i);
            if(curVal > curMax) {
                curMax = curVal;
                curMaxId = j;
            }
        }
        predictions.push_back(curMaxId);
    }
    return predictions;
}
float accuracy(Matrix output, std::vector<int> labels) {
    int numCorrect = 0;
    std::vector<int> predictions = getPredictions(output);
    for(int i = 0; i < labels.size(); i++) {
        if(predictions[i] == labels[i]) numCorrect++;
    }
    return (float) numCorrect/labels.size();
}


Trainer::Trainer(Network& n, DataSet d ) : network(n), dataset(d){
    std::vector<DataPoint> xValSubset;
    std::vector<int> yValSubset;
    for(int i =0; i < 1000; i++) {
        xValSubset.push_back(dataset.testData[i]);
        yValSubset.push_back(dataset.testData[i].label);
    }

    xVal = getMatrix(xValSubset);
    yVal = std::move(yValSubset);
}
void Trainer::train(int batchSize, int nEpochs, std::string networkSaveFile) {
    // create batches and loop through them
    int numBatches = dataset.trainData.size() / batchSize;
    for(int e = 0; e < nEpochs; e++) {
        for(int i = 0; i < numBatches; i++) {
            std::vector<DataPoint> batchData;
            for(int j =0; j < batchSize; j++) {
                batchData.push_back(dataset.trainData[i*batchSize + j]);
            }
            Matrix batch = getMatrix(batchData);
            std::vector<int> batchLabels;
            for(int j = 0; j < batchData.size(); j++) {
                batchLabels.push_back(batchData[j].label);
            }

            Matrix batchOutput = network.forward(batch);
            network.backpropagation(batch, batchLabels);
            std::vector<int> batchPredictions = getPredictions(batchOutput);
            float batchAcc = accuracy(batchOutput, batchLabels);
            float batchLoss = getBatchLoss(batchOutput, batchLabels);

            if(i == (numBatches - 1)) {
                Matrix valOutput = network.forward(xVal);
                float valAcc = accuracy(valOutput, yVal);
                float valLoss = getBatchLoss(valOutput, yVal);
                std::cout << "end epoch " << e << " with validation accuracy: " << valAcc << "; loss->" << valLoss << std::endl;
                if(e == (nEpochs - 1)){
                    // FIXME only way to save progress on a network is through storing and reading in a file.
                    network.saveToFile(networkSaveFile);
                }
            }

        }
    }
    

}

void testVerySimpleData() {
    std::vector<DataPoint> points;
    float feat;
    float feat2;
    int label;
    for(int i = 0; i <10000; i++) {
        feat = (float) (static_cast <float> (std::rand()) / static_cast <float> (RAND_MAX));
        label = feat > 0.5f ? 1 : 0;
        DataPoint point({{feat}, label});
        points.push_back(point);
    }
    DataSet dataset(points);
    Layer layer1(1, 8);
    layer1.activationFunctionId = sigmoid;
    Layer layer3(8, 2);
    layer3.activationFunctionId = softmax;
    Network network;
    network.layers.push_back(std::move(layer1));
    network.layers.push_back(std::move(layer3));

    Trainer trainer(network, dataset);
    trainer.train(5, 20, "verySimpleNetwork.dat");
    Matrix testInput(5, 1, {0.2f, 0.6f, 0.5f, 0.1f, 0.9f});
    Matrix result = network.forward(testInput);
    std::cout << "test matrix result:\n";
    std::cout << result << std::endl;

}

void testSpiralData() {
    DataSet spiralDataSet(100, 3);
    Layer layer1(2, 4);
    layer1.activationFunctionId = sigmoid;
    Layer layer2(4, 3);
    layer2.activationFunctionId = sigmoid;
    Layer layer3(3, 3);
    layer3.activationFunctionId = sigmoid;
    Layer layer4(3, 3);
    layer4.activationFunctionId = sigmoid;
    Layer layer5(3, 2);
    layer5.activationFunctionId = softmax;
    Network network;
    network.layers.push_back(layer1);
    network.layers.push_back(layer2);
    network.layers.push_back(layer3);
    network.layers.push_back(layer4);
    network.layers.push_back(layer5);
    Trainer trainer(network, spiralDataSet);
    trainer.train(5, 5, "spiralNetwork.dat");
}

/**
 * @brief Simple average pooling for square inputs for compression the number of features for learning MNIST.
 * returns a compressed version of the input by a factor of compressionFactor.
 * @param inputFeatures 
 * @param compressionFactor 
 * @return std::vector<float>
 */
std::vector<float> pool(std::vector<float> inputFeatures, int compressionFactor) {
    
    int originalSideLength = (int) std::sqrt(inputFeatures.size());
    int compressedSideLength = originalSideLength / compressionFactor;
    int curRow = 0;
    int curCol = 0;
    int convolutionArea = compressionFactor * compressionFactor;
    std::vector<float> result;
    for(int i = 0; i < compressedSideLength; i++) {
        for(int j = 0; j < compressedSideLength; j++) {
            float curSquareSum = 0;
            for(int k = i*compressionFactor; k < (i+1)*compressionFactor; k++){
                for(int l = j*compressionFactor; l < (j +1) * compressionFactor; l++) {
                    int index = k*originalSideLength + l;
                    curSquareSum += inputFeatures[index];
                }
            }
            result.push_back(curSquareSum / convolutionArea);
        }
    }
    return result;
}

void testMnistData() {
    std::ifstream mnistDataFile("mnist_x_train.dat");
    std::string line;
    std::vector<DataPoint> mnistData;
    int counter = 0;
    if(mnistDataFile.is_open()) {
        while(std::getline(mnistDataFile, line)) {
            if(counter == 10000) break;
            std::stringstream ss(line);
            std::istream_iterator<std::string> begin(ss);
            std::istream_iterator<std::string> end;
            std::vector<std::string> featStrings(begin, end);
            std::vector<float> feats(28*28);
            for(int i = 0; i < featStrings.size(); i++) {
                feats[i] = std::stof(featStrings[i]);
            }
            DataPoint newPoint;
            newPoint.features = feats;
            mnistData.push_back(newPoint);
            counter++;
        }
        mnistDataFile.close();
    }
    std::ifstream mnistLabelsFile("mnist_y_train.dat");
    std::vector<int> tmpLabels;
    if(mnistLabelsFile.is_open()) {
        while(std::getline(mnistLabelsFile, line)) {
            tmpLabels.push_back(std::stoi(line));
        }
        mnistLabelsFile.close();
    }
    for(int i= 0; i < mnistData.size(); i++) {
        mnistData[i].label = tmpLabels[i];
    }
    // Use pooled mnist data if you want to have fewer trainable features.
    // std::vector<DataPoint> pooledMnistData;
    // for(DataPoint dp : mnistData) {
    //     DataPoint pooledDp;
    //     pooledDp.features = pool(dp.features, 4);
    //     pooledDp.label = dp.label;
    //     pooledMnistData.push_back(std::move(pooledDp));
    // } 
    DataSet mnistDataSet(mnistData);
    // Matrix mnistMat = getMatrix(mnistDataSet.points);
    // std::cout << "mnist size\n" << mnistMat.nRow << ", " << mnistMat.nCol << std::endl;
    Layer layer1(784, 128);
    layer1.activationFunctionId = sigmoid;
    Layer layer2(128, 64);
    layer2.activationFunctionId = sigmoid;
    Layer layer3(64, 10);
    layer3.activationFunctionId = softmax;


    Network network;
    network.layers.push_back(std::move(layer1));
    network.layers.push_back(std::move(layer2));
    network.layers.push_back(std::move(layer3));
    Trainer trainer(network, mnistDataSet);
    trainer.train(1, 3, "mnistNetwork.dat");
}

void testReadingNetworkFromFile(std::string filename) {
    Network network = createNetworkFromFile(filename);
    for (Layer l : network.layers) {
        std::cout << l.weightMatrix.nRow << ", " << l.weightMatrix.nCol << ", " << l.weightMatrix.weights.size() << "\n";
    }
    std::ifstream mnistDataFile("mnist_x_train.dat");
    std::string line;
    std::vector<DataPoint> mnistData;
    int counter = 0;
    if(mnistDataFile.is_open()) {
        while(std::getline(mnistDataFile, line)) {
            if(counter == 10000) break;
            std::stringstream ss(line);
            std::istream_iterator<std::string> begin(ss);
            std::istream_iterator<std::string> end;
            std::vector<std::string> featStrings(begin, end);
            std::vector<float> feats(28*28);
            for(int i = 0; i < featStrings.size(); i++) {
                feats[i] = std::stof(featStrings[i]);
            }
            DataPoint newPoint;
            newPoint.features = feats;
            mnistData.push_back(newPoint);
            counter++;
        }
        mnistDataFile.close();
    }
    std::ifstream mnistLabelsFile("mnist_y_train.dat");
    std::vector<int> tmpLabels;
    if(mnistLabelsFile.is_open()) {
        while(std::getline(mnistLabelsFile, line)) {
            tmpLabels.push_back(std::stoi(line));
        }
        mnistLabelsFile.close();
    }
    for(int i= 0; i < mnistData.size(); i++) {
        mnistData[i].label = tmpLabels[i];
    }
    DataSet mnistDataSet(mnistData);
    Matrix input = getMatrix(mnistDataSet.testData);
    std::cout << "input size " << input.nRow << " " << input.nCol << std::endl;
    Matrix output = network.forward(input);
    float acc = accuracy(output, mnistDataSet.testLabels);
    std::cout << "accuracy was " << acc*100 << "%\n";
}

Matrix Matrix::transpose() {
    Matrix m;
    m.nRow = nCol;
    m.nCol = nRow;
    std::vector<float> tWeights;
    for(int i=0; i < nCol; i++) {
        Vector v = getCol(i);
        for(auto& val: v.weights) {
            tWeights.push_back(val);
        }
    }
    m.weights = tWeights;
    return m;
}

/**
 * @brief Add the values in matrix m to those in the current matrix in place.
 * 
 * @param m 
 */
void Matrix::add(Matrix& m) {
    if(m.nRow != nRow && m.nCol != nCol) {
        std::string errorStr = std::to_string(m.nRow) + "x" + std::to_string(m.nCol) + "!=" + std::to_string(nRow) + "x" + std::to_string(nCol); 
        throw std::invalid_argument("Matrix dimensionality error. Cannot add two matrices of unequal dimensions.");
    }
    for(int i=0; i < m.weights.size(); i++) {
        this->weights[i] += m.weights[i];
    }
}

std::vector<float> Layer::backward(Layer& leftLayer, int sampleIdx){
    /**
     * @brief Calculate the delta weights. Return local gradient values for next layer
     * 
     */
    std::vector<float> prevLayerOutput = leftLayer.previousOutput.getCol(sampleIdx).weights;
    //========== increment deltaWeights and deltaBiases for the layer
    Matrix deltaWeightsToIncrement(nodeLocalGradients, prevLayerOutput);
    deltaWeights.add(deltaWeightsToIncrement);
    for(int i = 0; i < deltaBiases.size(); i++) {
        deltaBiases[i] += nodeLocalGradients[i];
    }

    //========== Calculate the local gradients for the next layer in backpropagation
    std::vector<float> localGradientCopy;
    for(int i= 0; i < nodeLocalGradients.size(); i++) {
        localGradientCopy.push_back(nodeLocalGradients[i]);
    }
    Matrix localGradientMat(nodeLocalGradients.size(), 1, localGradientCopy);
    Matrix weight_T = weightMatrix.transpose();
    Matrix leftLayerLocalGradientsMat = weight_T * localGradientMat;

    std::vector<float> activationFunctionComponent;
    float curValue;
    switch(leftLayer.activationFunctionId) {
        case relu:
            for(int i = 0; i < prevLayerOutput.size(); i++) {
                curValue = prevLayerOutput[i] > 0 ? 1 : 0;
                activationFunctionComponent.push_back(curValue);
            }
            break;
        case sigmoid:
            for(int i = 0; i < prevLayerOutput.size(); i++) {
                curValue = prevLayerOutput[i] * (1 - prevLayerOutput[i]);
                activationFunctionComponent.push_back(curValue);
            } 
            break;
    }
    return pointwiseMult(leftLayerLocalGradientsMat.weights, activationFunctionComponent);
}

void Layer::updateWeightsAndBiases(int numSamples) {
    for(int i = 0; i < weightMatrix.nRow; i++) {
        for(int j = 0; j < weightMatrix.nCol; j++) {
            weightMatrix.setValAt(i, j, weightMatrix.getValAt(i, j) - LEARNING_RATE/numSamples * deltaWeights.getValAt(i, j));
        }
    }
    for(int i = 0; i < biases.size(); i++) {
        biases[i] -= LEARNING_RATE * deltaBiases[i] / numSamples;
    }
}


void Layer::setLocalGradients(std::vector<float> localGradients) {
    nodeLocalGradients = localGradients;
}

std::ostream& operator<<(std::ostream& out, Matrix &m){
    for(int i = 0; i < m.nRow; i++) {
        for(int j = 0; j < m.nCol; j++) {
            out << m.weights.at(i*m.nCol + j) << " ";
        }
        out << std::endl;
    }
    return out;
}

std::ostream& operator<<(std::ostream& out, Vector& v) {
    for(auto & x : v.weights) {
        out << x << ", ";
    }
    return out;
}
std::ostream& operator<<(std::ostream& out, Layer& l){
    out << "weights:\n" << l.weightMatrix << "\n\nbiases:" << std::endl;
    for(auto& val: l.biases) {
        out << val << std::endl;
    }
    return out;
}


Matrix operator*(Matrix& A, Matrix& B) {
    if(A.nCol != B.nRow) {
        std::cout << "matrix multiplication error: matrices don't have matching dimensions and cannot be multiplied" << std::endl;
        throw std::invalid_argument("matrix multiplication error: matrices don't have matching dimensions and cannot be multiplied");
    }
    std::vector<float> values;
    for(int i = 0; i < A.nRow; i++) {
        Vector curRow = A.getRow(i);
        for(int j = 0; j < B.nCol; j++) {
            Vector curCol = B.getCol(j);
            values.push_back(curRow.dot(curCol));
        }
    }
    return Matrix(A.nRow, B.nCol, values);
}

// int main() {    
//     // testVerySimpleData();
//     // testSpiralData();
//     // testMnistData();
//     testReadingNetworkFromFile("mnistNetwork.dat");

//     return 0;
// }