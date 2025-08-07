#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <map>

#include "transformer_types.h"
#include "self_attention.h"
#include "transformer_layers.h"
#include "tokenizer_embeddings.h"

// Main Transformer class
class Transformer {
private:
    Tokenizer tokenizer;
    Embeddings embeddings;
    Encoder encoder;
    Decoder decoder;
    Matrix outputLayerWeights; // For final prediction
    int embeddingDim;
    int vocabSize;

public:
    Transformer(int embedDim, int numHeads, int ffnHiddenDim, int numLayers, int maxSeqLen)
        : embeddings(1, embedDim, maxSeqLen), // Vocab size will be updated after tokenization
          encoder(numLayers, embedDim, numHeads, ffnHiddenDim),
          decoder(numLayers, embedDim, numHeads, ffnHiddenDim),
          embeddingDim(embedDim) {
        // Initialize output layer weights (vocab_size x embedding_dim)
        // Will be re-initialized after tokenizer builds vocabulary
    }

    void build(const std::vector<std::string>& corpus) {
        tokenizer.buildVocabulary(corpus);
        vocabSize = tokenizer.getVocabSize();
        
        // Re-initialize embeddings and output layer with correct vocab size
        embeddings = Embeddings(vocabSize, embeddingDim, embeddings.getMaxSequenceLength());
        Utils::initializeMatrix(outputLayerWeights, embeddingDim, vocabSize);
    }

    // Simplified prediction for a given input sequence
    std::string predictNextWord(const std::string& sentence) {
        std::istringstream iss(sentence);
        std::string word;
        std::vector<std::string> tokens;
        while (iss >> word) {
            tokens.push_back(word);
        }

        if (tokens.empty()) return "";

        // Prepare encoder input
        Matrix encoderInput(tokens.size(), Vector(embeddingDim));
        for (size_t i = 0; i < tokens.size(); ++i) {
            int tokenIdx = tokenizer.encode(tokens[i]);
            if (tokenIdx == -1) {
                std::cerr << "Warning: Unknown token \"" << tokens[i] << "\"\n";
                // Handle unknown tokens, e.g., by assigning a special <unk> embedding
                // For now, we'll just use a zero vector or skip.
                encoderInput[i].assign(embeddingDim, 0.0f);
            } else {
                encoderInput[i] = embeddings.getEmbedding(tokenIdx, i);
            }
        }

        // Run through encoder
        Matrix encoderOutput = encoder.forward(encoderInput);

        // Simplified decoder for next word prediction
        // We'll use the last encoder output as context for prediction
        // In a real scenario, the decoder would generate token by token.
        Vector lastEncoderOutput = encoderOutput[encoderOutput.size() - 1];

        // Apply output layer to get logits
        Vector logits = Utils::matMul(lastEncoderOutput, outputLayerWeights);

        // Apply softmax to get probabilities
        Vector probabilities = Utils::softmax(logits);

        // Find the word with the highest probability
        int predictedIndex = 0;
        float maxProb = 0.0f;
        for (size_t i = 0; i < probabilities.size(); ++i) {
            if (probabilities[i] > maxProb) {
                maxProb = probabilities[i];
                predictedIndex = i;
            }
        }

        return tokenizer.decode(predictedIndex);
    }
};

int main() {
    // Hyperparameters
    const int EMBEDDING_DIM = 64;
    const int NUM_HEADS = 4;
    const int FFN_HIDDEN_DIM = 128;
    const int NUM_LAYERS = 2;
    const int MAX_SEQ_LEN = 100;

    // Create and build transformer
    Transformer transformer(EMBEDDING_DIM, NUM_HEADS, FFN_HIDDEN_DIM, NUM_LAYERS, MAX_SEQ_LEN);

    // Example corpus for tokenizer (very small for demonstration)
    std::vector<std::string> corpus = {
        "the quick brown fox jumps over the lazy dog",
        "the dog barks loudly",
        "fox is a clever animal"
    };
    transformer.build(corpus);

    std::string sentence;
    std::cout << "Enter a sentence (e.g., \"the quick brown\"): ";
    std::getline(std::cin, sentence);

    std::string predictedWord = transformer.predictNextWord(sentence);
    std::cout << "Predicted next word: " << predictedWord << "\n";

    return 0;
}


