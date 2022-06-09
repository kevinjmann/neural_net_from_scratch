#include "network.hpp"
#include <vector>
#include <emscripten/bind.h>

Network network = createNetworkFromFile("./mnistNetwork.dat");

int doWasmInference(std::vector<float> input) {
    return network.doInference(input);
}

EMSCRIPTEN_BINDINGS(module) {
    emscripten::function("doWasmInference", &doWasmInference);
    emscripten::register_vector<float>("VectorFloat");
}
