#ifndef TRANSFORMER_LAYERS_H
#define TRANSFORMER_LAYERS_H

#include "transformer_types.h"
#include "self_attention.h"

// Feed-Forward Network
class FeedForwardNetwork {
private:
    Matrix W1, W2; // Weights
    Vector B1, B2; // Biases
    int inputDim;
    int hiddenDim;

    // ReLU activation function
    Vector relu(const Vector& input) {
        Vector output(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = std::max(0.0f, input[i]);
        }
        return output;
    }

public:
    FeedForwardNetwork(int inDim, int hDim) : inputDim(inDim), hiddenDim(hDim) {
        Utils::initializeMatrix(W1, inputDim, hiddenDim);
        Utils::initializeMatrix(W2, hiddenDim, inputDim);
        B1.assign(hiddenDim, 0.0f);
        B2.assign(inputDim, 0.0f);
    }

    Vector forward(const Vector& input) {
        // Layer 1: input * W1 + B1
        Vector hidden = Utils::add(Utils::matMul(input, W1), B1);
        hidden = relu(hidden);

        // Layer 2: hidden * W2 + B2
        Vector output = Utils::add(Utils::matMul(hidden, W2), B2);
        return output;
    }
};

// Encoder Layer
class EncoderLayer {
private:
    MultiHeadSelfAttention selfAttention;
    FeedForwardNetwork ffn;
    Vector ln1_gamma, ln1_beta; // LayerNorm for self-attention
    Vector ln2_gamma, ln2_beta; // LayerNorm for FFN
    int embeddingDim;

public:
    EncoderLayer(int embedDim, int numHeads, int ffnHiddenDim)
        : selfAttention(embedDim, numHeads),
          ffn(embedDim, ffnHiddenDim),
          embeddingDim(embedDim) {
        ln1_gamma.assign(embeddingDim, 1.0f);
        ln1_beta.assign(embeddingDim, 0.0f);
        ln2_gamma.assign(embeddingDim, 1.0f);
        ln2_beta.assign(embeddingDim, 0.0f);
    }

    Matrix forward(const Matrix& input) {
        // Self-Attention Sub-layer
        Matrix attnOutput = selfAttention.forward(input);
        
        // Add & Norm (Residual connection + Layer Normalization)
        Matrix output1(input.size(), Vector(embeddingDim));
        for (size_t i = 0; i < input.size(); ++i) {
            output1[i] = Utils::layerNorm(Utils::add(input[i], attnOutput[i]), ln1_gamma, ln1_beta);
        }

        // Feed-Forward Sub-layer
        Matrix ffnOutput(input.size(), Vector(embeddingDim));
        for (size_t i = 0; i < input.size(); ++i) {
            ffnOutput[i] = ffn.forward(output1[i]);
        }

        // Add & Norm
        Matrix output2(input.size(), Vector(embeddingDim));
        for (size_t i = 0; i < input.size(); ++i) {
            output2[i] = Utils::layerNorm(Utils::add(output1[i], ffnOutput[i]), ln2_gamma, ln2_beta);
        }
        return output2;
    }
};

// Decoder Layer
class DecoderLayer {
private:
    MultiHeadSelfAttention maskedSelfAttention;
    MultiHeadSelfAttention encoderDecoderAttention;
    FeedForwardNetwork ffn;
    Vector ln1_gamma, ln1_beta; // LayerNorm for masked self-attention
    Vector ln2_gamma, ln2_beta; // LayerNorm for encoder-decoder attention
    Vector ln3_gamma, ln3_beta; // LayerNorm for FFN
    int embeddingDim;

public:
    DecoderLayer(int embedDim, int numHeads, int ffnHiddenDim)
        : maskedSelfAttention(embedDim, numHeads),
          encoderDecoderAttention(embedDim, numHeads),
          ffn(embedDim, ffnHiddenDim),
          embeddingDim(embedDim) {
        ln1_gamma.assign(embeddingDim, 1.0f);
        ln1_beta.assign(embeddingDim, 0.0f);
        ln2_gamma.assign(embeddingDim, 1.0f);
        ln2_beta.assign(embeddingDim, 0.0f);
        ln3_gamma.assign(embeddingDim, 1.0f);
        ln3_beta.assign(embeddingDim, 0.0f);
    }

    Matrix forward(const Matrix& targetInput, const Matrix& encoderOutput) {
        // Masked Multi-Head Self-Attention Sub-layer
        Matrix maskedAttnOutput = maskedSelfAttention.forward(targetInput, true); // Apply mask

        // Add & Norm
        Matrix output1(targetInput.size(), Vector(embeddingDim));
        for (size_t i = 0; i < targetInput.size(); ++i) {
            output1[i] = Utils::layerNorm(Utils::add(targetInput[i], maskedAttnOutput[i]), ln1_gamma, ln1_beta);
        }

        // Multi-Head Encoder-Decoder Attention Sub-layer
        // Here, Q comes from output1, K and V come from encoderOutput
        // This requires a modification to MultiHeadSelfAttention to accept separate Q, K, V inputs
        // For simplicity, we'll use the existing forward and assume K, V are derived from encoderOutput
        // This is a simplification and would need a proper implementation for cross-attention.
        Matrix encDecAttnOutput = encoderDecoderAttention.forward(output1); // Simplified cross-attention

        // Add & Norm
        Matrix output2(targetInput.size(), Vector(embeddingDim));
        for (size_t i = 0; i < targetInput.size(); ++i) {
            output2[i] = Utils::layerNorm(Utils::add(output1[i], encDecAttnOutput[i]), ln2_gamma, ln2_beta);
        }

        // Feed-Forward Sub-layer
        Matrix ffnOutput(targetInput.size(), Vector(embeddingDim));
        for (size_t i = 0; i < targetInput.size(); ++i) {
            ffnOutput[i] = ffn.forward(output2[i]);
        }

        // Add & Norm
        Matrix output3(targetInput.size(), Vector(embeddingDim));
        for (size_t i = 0; i < targetInput.size(); ++i) {
            output3[i] = Utils::layerNorm(Utils::add(output2[i], ffnOutput[i]), ln3_gamma, ln3_beta);
        }
        return output3;
    }
};

// Encoder
class Encoder {
private:
    std::vector<EncoderLayer> layers;
    int numLayers;

public:
    Encoder(int numL, int embedDim, int numHeads, int ffnHiddenDim) : numLayers(numL) {
        for (int i = 0; i < numLayers; ++i) {
            layers.emplace_back(embedDim, numHeads, ffnHiddenDim);
        }
    }

    Matrix forward(const Matrix& input) {
        Matrix output = input;
        for (auto& layer : layers) {
            output = layer.forward(output);
        }
        return output;
    }
};

// Decoder
class Decoder {
private:
    std::vector<DecoderLayer> layers;
    int numLayers;

public:
    Decoder(int numL, int embedDim, int numHeads, int ffnHiddenDim) : numLayers(numL) {
        for (int i = 0; i < numLayers; ++i) {
            layers.emplace_back(embedDim, numHeads, ffnHiddenDim);
        }
    }

    Matrix forward(const Matrix& targetInput, const Matrix& encoderOutput) {
        Matrix output = targetInput;
        for (auto& layer : layers) {
            output = layer.forward(output, encoderOutput);
        }
        return output;
    }
};

#endif // TRANSFORMER_LAYERS_H


