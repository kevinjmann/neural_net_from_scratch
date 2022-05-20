#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include <fstream>

enum activationFunction {
    relu,
    sigmoid,
    step,
    nothing,
    softmax
};
struct DataPoint {
    float x;
    float y;
    friend std::ostream& operator<<(std::ostream& out, DataPoint &dp) {
        out <<dp.x << ", " << dp.y;
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
            std::cout << "lengths do not match, dot product could not be calculated" << std::endl;
            return 0;
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
    Matrix(){}
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
                weights.push_back(vec1[i] * vec2[j]);
            }
        }
    }
    Matrix(const Matrix &m) {
        nCol = m.nCol;
        nRow = m.nRow;
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
        for(int i = 0; i < weights.size(); i++) {
            m.weights.push_back(weights[i]);
        }
        return m;
    }
    Vector getRow(int i) {
        if(i > nRow || i < 0) {
            std::cout << "invalid row index" << std::endl;
            // TODO missing throw exception
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
            std::cout << "invalid column id" << std::endl;
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
class DataSet {
    public:
    std::vector<DataPoint> points;
    std::vector<int> classes;
    std::vector<int> labels;
    DataSet(int numPoints, int numC){
        for(int i = 0; i < numC; i++) {
            classes.push_back(i);
        }
        int ix;
        for(int i = 0; i < numC; i++) {
            for(int ix = 0; ix < numPoints; ix++) {
                float r = (float) (ix) / numPoints;
                // std::cout << "r: " << r << std::endl;
                float t = (float)(((i+1)*4 - (i * 4)) *(i+1)* ix)/numPoints + static_cast <float> (std::rand()) / static_cast <float> (RAND_MAX) * 0.5f;
                // std:: cout << "t: " << t << std::endl;
                DataPoint dp({(float) (r*sin(t*2.5)), (float)(r*cos(t*2.5))});
                points.push_back(dp);
                labels.push_back(i);
            }
        }
    }
    Matrix getMatrix() {
        std::vector<float> weights;
        for(int i = 0; i < points.size(); i++) {
            weights.push_back(points[i].x);
            weights.push_back(points[i].y);
        }
        Matrix m(points.size(), 2, weights);
        return m;
    }
};

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
    float activation(float& val){
        float res;
        switch(activationFunctionId) {
            case relu: res = val > 0 ? val : 0; break;
            case step: res = val > 0 ? 1: 0; break;
            case sigmoid: res = 1/(1 + std::exp(val)); break;
            default: res = val; break;
        }
        return res;
    }

    void activateRow(float* val, int numVals) {
        float totalExp = 0;
        float rowMax =0;
        if(activationFunctionId == softmax){
            for(int i=0; i < numVals; i++) {
                if(val[i] > rowMax) {
                    rowMax = val[i];
                }
            }
            for(int i = 0; i < numVals; i++) {
                val[i]-=rowMax;
                totalExp += std::exp(val[i]);
            }
        }
        switch(activationFunctionId) {
            case relu:
                for(int i = 0; i < numVals; i++) {
                    val[i] = val[i] > 0 ? val[i] : 0;
                }
                break;
            case step:
                for(int i=0; i < numVals; i++) {
                    val[i] = val[i] > 0 ? 1 : 0;
                }
                break;
            case sigmoid:
                for(int i=0; i < numVals; i++) {
                    val[i] = 1/(1 + std::exp(val[i]));
                }
                break;
            case softmax:
                for(int i=0; i < numVals; i++) {
                    val[i] = std::exp(val[i]) / totalExp;
                }
                break;

            default: break;
        }
        void updateWeights();
    }
    
    Matrix forward(Matrix &input) {
        previousOutput = input*weights;
        for(int i = 0; i <previousOutput.nRow; i++) {
            for(int j = 0; j < previousOutput.nCol; j++) {
                previousOutput.setValAt(i,j, previousOutput.getValAt(i, j) + biases[j]);
            }
            // FIXME add in summation to accumulator variables to take average after this loop.
            activateRow(&(previousOutput.weights[i*weights.nCol]), weights.nCol);

            
        }
        return previousOutput;
    }

    std::vector<float> backward(Layer& leftLayer);
    void setLocalGradients(std::vector<float> localGradients);
    void Layer::updateWeightsAndBiases();

    friend std::ostream& operator<<(std::ostream& out, Layer& l);
};

class Network {
    public:
    std::vector<Layer> layers;
    std::vector<Matrix> deltas;
    void forward(Matrix input) {
        layers[0].forward(input);
        for(int l = 1; l < layers.size(); l++) {
            layers[l].forward(layers[l-1].previousOutput);
        }
    }
    void backward(std::vector<int> labels) {
        std::vector<float> crossEntropyDeltas = getDeltaCrossEntropy(layers[layers.size() - 1].previousOutput, labels);
        int depth = 0;
        for(int l = layers.size() - 1; l >= 0; l--) {
            
            depth++;
        }
    }

};
const float eps = 1e-7;
float clipVal(float val) {
    if(val <= 0) return eps;
    if(val >=(1-eps)) return 1-eps;
    return val;
}


std::vector<float> getBatchLoss(Matrix output, std::vector<float> labels) {
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
float accuracy(Matrix output, std::vector<float> labels) {
    int numCorrect = 0;
    std::vector<int> predictions = getPredictions(output);
    for(int i = 0; i < labels.size(); i++) {
        if(predictions[i] == labels[i]) numCorrect++;
    }
    return (float) numCorrect/labels.size();
}
std::vector<float> getDeltaCrossEntropy(Matrix softmaxOutputs, std::vector<int> labels) {
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
int main() {
    // std::vector<float> inputs = {
    //     1, 2, 3, 2.5,
    //     2, 5, -1, 2,
    //     -1.5, 2.7, 3.3, -0.8
    // };
    // Matrix input(3, 4, inputs);
    // std::vector<float> weights {
    //      0.2,  0.5,   -0.26,
    //      0.8, -0.91,  -0.27,
    //     -0.5,  0.26,   0.17,
    //      1,    -0.5,   0.87
    // };
    // Matrix m(4, 3, weights);
    // std::vector<float> biases{2, 3, 0.5};
    // Layer firstLayer(m, biases);
    // std::vector<float>l2weights {
    //     0.1, -0.5, -0.44,
    //     -0.14, 0.12, 0.73, 
    //     0.5, -0.33, -0.13
    // };
    // Matrix l2mat(3, 3, l2weights);
    // std::vector<float> biases2{-1, 2, -0.5};
    // Layer secondLayer(l2mat, biases2);
    // Layer randomLayer(5,3);
    // Matrix secondOutput = secondLayer.forward(firstLayer.forward(input));
    // std::cout << "2nd layer output\n" << secondOutput << std::endl;
    // std::cout << "random layer:\n" << randomLayer << std::endl;
    DataSet test(100, 3);
    Layer layer1(2, 3);
    layer1.activationFunctionId = relu;
    Layer layer2(3, 3);
    layer2.activationFunctionId = softmax;
    Matrix dataSetMat = test.getMatrix();
    // std::cout << dataSetMat << std::endl;
    Matrix l1out = layer1.forward(dataSetMat);
    Matrix l2out = layer2.forward(l1out);
    Network network;
    network.layers.push_back(layer1);
    network.layers.push_back(layer2);
    network.forward(dataSetMat);
    std::cout << l2out << "\n\n\n" << std::endl;
    std::cout << network.layers[1].previousOutput << "\n\n\n" << std::endl;
    std::vector<float> dce = getDeltaCrossEntropy(network.layers[1].previousOutput, test.labels);
    for(int i = 0; i < dce.size(); i++) {
        std::cout << dce[i] << std::endl;
    }
    // Matrix sampleMatrix(1, 3, {0.7, 0.1, 0.2});
    // std::vector<float> sampleLabels{0};
    // std::cout << "mean loss: "<< meanLoss(getBatchLoss(sampleMatrix, sampleLabels)) << std::endl;
    // // std::cout << l2out << std::endl;



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
    Matrix deltaWeightsToSet(leftLayer.avgActivatedOutput, nodeLocalGradients);
    deltaWeights = deltaWeightsToSet;
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
            for(int i = 0; i < leftLayer.avgUnactivatedOutput.size(); i++) {
                curValue = leftLayer.avgUnactivatedOutput[i] > 0 ? 1 : 0;
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
            weights.setValAt(i, j, weights.getValAt(i, j) - deltaWeights.getValAt(i, j));
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
        std::cout << "matrices don't have matching dimensions and cannot be multiplied" << std::endl;
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

std::vector<float> pointwiseMult(std::vector<float> vecA, std::vector<float> vecB) {
    std::vector<float> result;
    if(vecA.size() != vecB.size()) {
        std::cout << "pointwise multiplication error. Both vectors must be of equal size." << std::endl;
        return result;
    }
    for(int i = 0; i < vecA.size(); i++) {
        result.push_back(vecA[i] * vecB[i]);
    }
    return result;
}