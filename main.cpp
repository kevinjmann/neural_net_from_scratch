#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include <fstream>
#include <algorithm>
#include <random>
#include <sstream>
#include <iterator>

std::vector<float> pointwiseMult(std::vector<float> vecA, std::vector<float> vecB) {
    /**
     * @brief Multiply two vectors pointwise and recieve a new vector of the same size.
     * 
     */
    std::vector<float> result;
    if(vecA.size() != vecB.size()) {
        throw "pointwiseMult: pointwise multiplication error. Both vectors must be of equal size.";
    }
    for(int i = 0; i < vecA.size(); i++) {
        result.push_back(vecA[i] * vecB[i]);
    }
    return result;
}

const float LEARNING_RATE = 1;

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
    friend std::ostream& operator<<(std::ostream& out, DataPoint &dp) {
        int counter = 0;
        for(auto feature : dp.features) {
            if (counter < 10 || counter > (dp.features.size() - 10)) {
                out << feature << " ";
            }
            if(counter == 10) {
                out << " ... ";
            }
            counter++;
        }
        out << "\nclass: " << dp.label;
        return out; 
    }
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
    float dot(Vector a) {
        if(length != a.length) {
            throw "lengths do not match, dot product could not be calculated";
        }
        float res = 0;
        for(int i = 0; i < length; i++) {
            res += this->weights[i] * a.weights[i];
        }
        return res;
    }
    friend std::ostream& operator<<(std::ostream& out, Vector& v);
};
class Matrix {
    public:
    int nCol;
    int nRow;
    std::vector<float> weights;
    Matrix(){
        nCol = 0;
        nRow = 0;
    }
    Matrix(int r, int c) : nCol(c), nRow(r) {
        for(int i = 0; i < c*r; i++) {
            weights.push_back(0);
        }
    }
    Matrix(int r, int c, std::vector<float> w) : nCol(c), nRow(r), weights(w) {
    }
    Matrix(std::vector<float> vec1, std::vector<float> vec2) {
        /**
         * @brief Create a matrix from two vectors by multipling them into a grid. Used to calculate delta weight values.
         * 
         */
        nCol = vec1.size();
        nRow = vec2.size();
        for (int i = 0; i < nRow; i++) {
            for (int j = 0; j < nCol; j++) {
                weights.push_back(vec1[j] * vec2[i]);
            }
        }
    }
    Matrix(const Matrix &m) {
        nCol = m.nCol;
        nRow = m.nRow;
        weights = {};
        for(int i=0; i < m.nRow; i++) {
            for(int j = 0; j < m.nCol; j++)  {
                weights.push_back(m.getValAt(i, j));
            }
        }
    }
    Matrix copy() {
        Matrix m;
        m.nCol = nCol;
        m.nRow = nRow;
        m.weights = {};
        for(int i = 0; i < weights.size(); i++) {
            m.weights.push_back(weights[i]);
        }
        return m;
    }
    Vector getRow(int i) {
        if(i > nRow || i < 0) {
            throw "getRow: invalid row index";
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
    Vector getCol(int i) {
        if(i > nCol || i < 0) {
            throw "getCol: invalid colummn id";
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

    void setValAt(int r, int c, float val) {
        weights[r*nCol + c] = val;
    }
    float getValAt(int r, int c) const {
        return weights[r*nCol + c];
    }

    void transpose();
    friend std::ostream& operator<<(std::ostream& out, Matrix &m);
    friend Matrix operator*(Matrix& A, Matrix& B);
};

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

std::vector<float> getDeltaCrossEntropy(Matrix softmaxOutputs, std::vector<int> labels) {
    /**
     * @brief Calculates delta values for the output layer.
     * 
     * 
     */
    Matrix batchDeltas(softmaxOutputs);
    std::vector<float> result;
    for(int i = 0; i < softmaxOutputs.nCol; i++) {
        result.push_back(0);
    }

    for(int i = 0; i < labels.size(); i++) {
        batchDeltas.setValAt(i, labels[i], batchDeltas.getValAt(i, labels[i]) -1);
        for (int j = 0; j < batchDeltas.nCol; j++) {
            result[j] += batchDeltas.getValAt(i, j);
        }
    }
    // take average across batch
    for(int i = 0; i < result.size(); i++) {
        result[i] /= labels.size();
    }
    return result;
}

class DataSet {
    public:
    std::vector<DataPoint> points;
    std::vector<int> classes; // contains all the possible classes.
    std::vector<DataPoint> trainData;
    std::vector<int> trainLabels;

    std::vector<DataPoint> testData;
    std::vector<int> testLabels;
    float trainTestSplit = 0.9;
    DataSet(int numPoints, int numC){
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
    DataSet(std::vector<DataPoint> p) : points(p){
        initializeTrainTestData();
    } 
    void shuffleData (){
        /**
         * @brief Shuffle the data points in place
         * cf https://stackoverflow.com/questions/6926433/how-to-shuffle-a-stdvector
         */
        auto rng = std::default_random_engine {};
        std::shuffle(std::begin(points), std::end(points), rng);
    }
    void initializeTrainTestData() {
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

    std::vector<int> getLabels() {
        std::vector<int> toReturn;
        for(int i = 0; i < points.size(); i++) {
            toReturn.push_back(points[i].label);
        }
        return toReturn;
    }
    static Matrix getMatrix(std::vector<DataPoint> pointSubset) {
        std::vector<float> weights;
        for(int i = 0; i < pointSubset.size(); i++) {
            for(int j =0; j < pointSubset[i].features.size(); j++) {
                weights.push_back(pointSubset[i].features[j]);
            }
        }
        Matrix m(pointSubset.size(), pointSubset[0].features.size(), weights);
        return m;
    }
};

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

class Layer {
    std::vector<float> biases;
    public:
    Matrix weights;
    Matrix deltaWeights;
    Matrix previousOutput;
    activationFunction activationFunctionId;

    // values used in backpropagation
    std::vector<float> avgUnactivatedOutput;
    std::vector<float> avgActivatedOutput;
    std::vector<float> nodeLocalGradients; // Note, also serves as a delta value for biases.

    Layer(int numFeatures, int numNeurons) {
        std::vector<float> initialWeights;
        for(int i = 0; i < numFeatures * numNeurons; i++) {
            float r = static_cast <float> (std::rand()) / static_cast <float> (RAND_MAX);
            initialWeights.push_back(r);
        }
        std::vector<float> resBiases;
        for(int i =0; i < numNeurons; i++) {
            // float r = static_cast <float> (std::rand()) / static_cast <float> (RAND_MAX);
            resBiases.push_back(0);
        }
        Matrix m(numFeatures, numNeurons, initialWeights);
        this->weights = m;
        this->biases = resBiases;
        activationFunctionId = step;
    }
    Layer(Matrix m, std::vector<float> b, activationFunction fId=step): weights(m), biases(b), activationFunctionId(fId){

    }

    std::vector<float> activateRow(std::vector<float> row) {
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
    
    Matrix forward(Matrix &input) {
        /**
         * @brief Run input through a forward pass of the layer. Keep track of the avg. output for each node for later backpropagation.
         * 
         */
        this->previousOutput = input*weights;
        std::vector<float> avgOutputAccumulator;
        float inv_batch_size = (float) 1 / previousOutput.nRow;
        for(int i= 0; i < previousOutput.nCol; i++) {
            avgOutputAccumulator.push_back(0);
        }

        for(int i = 0; i <previousOutput.nRow; i++) {
            for(int j = 0; j < previousOutput.nCol; j++) {
                this->previousOutput.setValAt(i,j, this->previousOutput.getValAt(i, j) + this->biases[j]);
            }
            std::vector<float> activatedRow = activateRow(this->previousOutput.getRow(i).weights);
            for(int j =0; j < activatedRow.size(); j++) {
                this->previousOutput.setValAt(i, j, activatedRow[j]);
            }
            for(int j = 0; j < this->weights.nCol; j++) {
                avgOutputAccumulator[j] += this->previousOutput.weights[i*this->weights.nCol + j];
            }
            
        }
        for(int i =0; i < avgOutputAccumulator.size(); i++) {
            avgOutputAccumulator[i] = avgOutputAccumulator[i] / input.nRow;
        }

        avgActivatedOutput = avgOutputAccumulator;
        return previousOutput;
    }

    std::vector<float> backward(Layer& leftLayer);
    void setLocalGradients(std::vector<float> localGradients);
    void updateWeightsAndBiases();

    friend std::ostream& operator<<(std::ostream& out, Layer& l);
}; // class Layer

class Network {
    public:
    std::vector<Layer> layers;
    std::vector<Matrix> deltas;
    std::vector<float> avgBatchInputs;
    Matrix forward(Matrix input) {
        layers[0].forward(input);
        for(int l = 1; l < layers.size(); l++) {
            layers[l].forward(layers[l-1].previousOutput);
        }
        avgBatchInputs = getFeatureWiseAverage(input);
        return layers[layers.size() - 1].previousOutput;
    }
    void backward(std::vector<int> labels) {
        
        std::vector<float> crossEntropyDeltas = getDeltaCrossEntropy(layers[layers.size() - 1].previousOutput, labels);
        // set local gradients for output layer.
        layers[layers.size() -1].setLocalGradients(crossEntropyDeltas);
        // handle layers besides input layer
        for(int l = layers.size() - 1; l > 0; l--) {
            // TODO clip these local gradients to mitigate exploding gradient.
            std::vector<float> l_1_local_gradients = layers[l].backward(layers[l-1]);
            layers[l-1].setLocalGradients(l_1_local_gradients);
        }

        // handle input layer
        Matrix inputLayerDeltaWeights(layers[0].nodeLocalGradients, avgBatchInputs);
        layers[0].deltaWeights = inputLayerDeltaWeights;
        for(int l =0; l < layers.size(); l++) {
            layers[l].updateWeightsAndBiases();
        }
    }

}; // class Network
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

std::vector<float> getBatchLoss(Matrix output, std::vector<int> labels) {
    std::vector<float> losses;
    for(int i = 0; i < output.nRow; i++) {
        losses.push_back(
            -std::log(
                clipVal(output.getValAt(i, labels[i]))));
    }
    return losses;
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
    for(int i = 0; i < output.nRow; i++){
        float curMax = 0;
        int curMaxId = 0;
        float curVal;
        for(int j =0; j < output.nCol; j++) {
            curVal = output.getValAt(i, j);
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

class Trainer {
    public:
    Network network;
    DataSet dataset;
    Trainer(Network n, DataSet d ) : network(n), dataset(d){}
    void train(int batchSize, int nEpochs) {
        // create batches and loop through them
        int numBatches = dataset.trainData.size() / batchSize;
        float epochAccuracy;
        float epochLoss;
        for(int e = 0; e < nEpochs; e++) {
            epochAccuracy = 0;
            epochLoss = 0;
            for(int i = 0; i < numBatches; i++) {
                std::vector<DataPoint> batchData;
                for(int j =0; j < batchSize; j++) {
                    batchData.push_back(dataset.trainData[i*batchSize + j]);
                }
                Matrix batch = DataSet::getMatrix(batchData);
                std::vector<int> batchLabels;
                for(int j = 0; j < batchData.size(); j++) {
                    batchLabels.push_back(batchData[j].label);
                }
                // FIXME updating weights for batches should be done with the sum of the deltas for all samples in the batch.
                // Currently, you average together the inputs and deltas for a whole batch which is incorrect.
                Matrix batchOutput = network.forward(batch);
                network.backward(batchLabels);

                // if (i == (numBatches - 1)) {
                    float batchAcc = accuracy(batchOutput, batchLabels);
                    float batchLoss = meanLoss(getBatchLoss(batchOutput, batchLabels));
                    epochAccuracy += batchAcc;
                    epochLoss += batchLoss;
                    // std::cout << "training loss for batch: " << meanLoss(getBatchLoss(batchOutput, batchLabels)) << std::endl;
                    // std::cout << "training accuracy for batch: " << accuracy(batchOutput, batchLabels) << std::endl;
                // }
                if(i == (numBatches - 1)) {
                    std::cout << "end epoch " << e << " with accuracy: " << epochAccuracy << "/" << dataset.trainData.size() << "; loss->" << epochLoss << std::endl; 
                }

            }
        }
        

    }
};

void testVerySimpleData() {
    std::vector<DataPoint> points;
    float feat;
    float feat2;
    int label;
    for(int i = 0; i <1000; i++) {
        feat = (float) (static_cast <float> (std::rand()) / static_cast <float> (RAND_MAX));
        label = feat > 0.5f ? 1 : 0;
        DataPoint point({{feat}, label});
        points.push_back(point);
    }
    DataSet dataset(points);
    Layer layer1(1, 2);
    layer1.activationFunctionId = sigmoid;
    Layer layer2(2, 2);
    layer2.activationFunctionId = sigmoid;
    Layer layer3(2, 2);
    layer3.activationFunctionId = softmax;
    Network network;
    network.layers.push_back(std::move(layer1));
    network.layers.push_back(std::move(layer2));
    network.layers.push_back(std::move(layer3));

    Trainer trainer(network, dataset);
    trainer.train(1, 1000);

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
    trainer.train(5, 5);
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
    std::ifstream mnistDataFile("mnist_data.dat");
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
    std::ifstream mnistLabelsFile("mnist_labels.dat");
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
    std::vector<DataPoint> convolutedMnistData;
    for(DataPoint dp : mnistData) {
        DataPoint convDp;
        convDp.features = pool(dp.features, 7);
        convDp.label = dp.label;
        convolutedMnistData.push_back(std::move(convDp));
    } 
    DataSet mnistDataSet(convolutedMnistData);
    Matrix mnistMat = DataSet::getMatrix(mnistDataSet.points);
    std::cout << "mnist size\n" << mnistMat.nRow << ", " << mnistMat.nCol << std::endl; 
    Layer layer1(16, 8);
    layer1.activationFunctionId = sigmoid;

    Layer layer2(8, 10);
    layer2.activationFunctionId = softmax;

    Network network;
    network.layers.push_back(std::move(layer1));
    network.layers.push_back(std::move(layer2));
    Trainer trainer(network, mnistDataSet);
    trainer.train(1, 1000);
}

int main() {    
    testVerySimpleData();
    // testSpiralData();
    // testMnistData();


    return 0;
}

void Matrix::transpose() {
    std::vector<float> tWeights;
    for(int i=0; i < nCol; i++) {
        Vector v = getCol(i);
        for(auto& val: v.weights) {
            tWeights.push_back(val);
        }
    }
    
    weights = tWeights;
    int temp = nRow;
    nRow = nCol;
    nCol = temp;
}

std::vector<float> Layer::backward(Layer& leftLayer){
    /**
     * @brief Calculate the delta weights. Return local gradient values for next layer
     * 
     */
    Matrix deltaWeightsToSet(nodeLocalGradients, leftLayer.avgActivatedOutput);
    // TODO get rid of needless copying
    deltaWeights = Matrix(deltaWeightsToSet);
    std::vector<float> localGradientCopy;
    for(int i= 0; i < nodeLocalGradients.size(); i++) {
        localGradientCopy.push_back(nodeLocalGradients[i]);
    }
    Matrix localGradientMat(nodeLocalGradients.size(), 1, localGradientCopy);

    Matrix leftLayerLocalGradientsMat = weights * localGradientMat;

    std::vector<float> activationFunctionComponent;
    float curValue;
    switch(leftLayer.activationFunctionId) {
        case relu:
            for(int i = 0; i < leftLayer.avgActivatedOutput.size(); i++) {
                curValue = leftLayer.avgActivatedOutput[i] > 0 ? 1 : 0;
                activationFunctionComponent.push_back(curValue);
            }
            break;
        case sigmoid:
            for(int i = 0; i < leftLayer.avgActivatedOutput.size(); i++) {
                curValue = leftLayer.avgActivatedOutput[i] * (1 - leftLayer.avgActivatedOutput[i]);
                activationFunctionComponent.push_back(curValue);
            } 
            break;
    }
    return pointwiseMult(leftLayerLocalGradientsMat.weights, activationFunctionComponent);
}

void Layer::updateWeightsAndBiases() {
    for(int i = 0; i < weights.nRow; i++) {
        for(int j = 0; j < weights.nCol; j++) {
            weights.setValAt(i, j, weights.getValAt(i, j) - LEARNING_RATE * deltaWeights.getValAt(i, j));
        }
    }
    for(int i = 0; i < biases.size(); i++) {
        biases[i] -= nodeLocalGradients[i];
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
    out << "weights:\n" << l.weights << "\n\nbiases:" << std::endl;
    for(auto& val: l.biases) {
        out << val << std::endl;
    }
    return out;
}


Matrix operator*(Matrix& A, Matrix& B) {
    if(A.nCol != B.nRow) {
        std::cout << "matrix multiplication error: matrices don't have matching dimensions and cannot be multiplied" << std::endl;
        throw "matrix multiplication error: matrices don't have matching dimensions and cannot be multiplied";
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

