#ifndef SELF_ATTENTION_H
#define SELF_ATTENTION_H

#include "transformer_types.h"

class MultiHeadSelfAttention {
private:
    int embeddingDim;
    int numHeads;
    int headDim;

    Matrix W_Q, W_K, W_V, W_O; // Weight matrices for Query, Key, Value, and Output

    // Helper function for scaled dot-product attention
    // Q, K, V are for a single head, and are matrices (seq_len x head_dim)
    Matrix scaledDotProductAttention(const Matrix& Q, const Matrix& K, const Matrix& V, bool mask = false) {
        // (Q * K^T) / sqrt(head_dim)
        Matrix scores(Q.size(), Vector(K.size()));
        for (size_t i = 0; i < Q.size(); ++i) {
            for (size_t j = 0; j < K.size(); ++j) {
                scores[i][j] = Utils::dotProduct(Q[i], K[j]) / std::sqrt(headDim);
            }
        }

        // Apply masking for decoder self-attention
        if (mask) {
            for (size_t i = 0; i < scores.size(); ++i) {
                for (size_t j = i + 1; j < scores[i].size(); ++j) {
                    scores[i][j] = -1e9; // Set to a very small number for masking
                }
            }
        }

        // Apply softmax to scores
        Matrix attentionWeights(scores.size(), Vector(scores[0].size()));
        for (size_t i = 0; i < scores.size(); ++i) {
            attentionWeights[i] = Utils::softmax(scores[i]);
        }

        // attentionWeights * V
        Matrix output(attentionWeights.size(), Vector(V[0].size(), 0.0f));
        for (size_t i = 0; i < attentionWeights.size(); ++i) { // For each query token
            for (size_t k = 0; k < V[0].size(); ++k) { // For each dimension in V
                for (size_t j = 0; j < attentionWeights[0].size(); ++j) { // For each key token
                    output[i][k] += attentionWeights[i][j] * V[j][k];
                }
            }
        }
        return output;
    }

public:
    MultiHeadSelfAttention(int embedDim, int nHeads) 
        : embeddingDim(embedDim), numHeads(nHeads), headDim(embedDim / nHeads) {
        
        // Initialize weight matrices
        Utils::initializeMatrix(W_Q, embeddingDim, embeddingDim);
        Utils::initializeMatrix(W_K, embeddingDim, embeddingDim);
        Utils::initializeMatrix(W_V, embeddingDim, embeddingDim);
        Utils::initializeMatrix(W_O, embeddingDim, embeddingDim);
    }

    // input: [seq_len, embeddingDim]
    Matrix forward(const Matrix& input, bool mask = false) {
        int seqLen = input.size();

        // Linear transformations for Q, K, V for all tokens in the sequence
        Matrix Q_all(seqLen, Vector(embeddingDim));
        Matrix K_all(seqLen, Vector(embeddingDim));
        Matrix V_all(seqLen, Vector(embeddingDim));

        for (int i = 0; i < seqLen; ++i) {
            Q_all[i] = Utils::matMul(input[i], W_Q);
            K_all[i] = Utils::matMul(input[i], W_K);
            V_all[i] = Utils::matMul(input[i], W_V);
        }

        // Split into multiple heads and compute attention
        Matrix concatenatedHeads(seqLen, Vector(embeddingDim));

        for (int h = 0; h < numHeads; ++h) {
            Matrix Q_head(seqLen, Vector(headDim));
            Matrix K_head(seqLen, Vector(headDim));
            Matrix V_head(seqLen, Vector(headDim));

            for (int i = 0; i < seqLen; ++i) {
                for (int d = 0; d < headDim; ++d) {
                    Q_head[i][d] = Q_all[i][h * headDim + d];
                    K_head[i][d] = K_all[i][h * headDim + d];
                    V_head[i][d] = V_all[i][h * headDim + d];
                }
            }

            Matrix headOutput = scaledDotProductAttention(Q_head, K_head, V_head, mask);

            // Concatenate heads
            for (int i = 0; i < seqLen; ++i) {
                for (int d = 0; d < headDim; ++d) {
                    concatenatedHeads[i][h * headDim + d] = headOutput[i][d];
                }
            }
        }

        // Final linear layer
        Matrix output(seqLen, Vector(embeddingDim));
        for (int i = 0; i < seqLen; ++i) {
            output[i] = Utils::matMul(concatenatedHeads[i], W_O);
        }

        return output;
    }
};

#endif // SELF_ATTENTION_H


