#ifndef TOKENIZER_EMBEDDINGS_H
#define TOKENIZER_EMBEDDINGS_H

#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <algorithm>
#include <cmath>
#include "transformer_types.h"

// Tokenizer class
class Tokenizer {
private:
    std::map<std::string, int> wordToIndex;
    std::map<int, std::string> indexToWord;
    int vocabSize;

public:
    Tokenizer() : vocabSize(0) {}

    void buildVocabulary(const std::vector<std::string>& sentences) {
        for (const auto& sentence : sentences) {
            std::istringstream iss(sentence);
            std::string word;
            while (iss >> word) {
                // Convert to lowercase for simplicity
                std::transform(word.begin(), word.end(), word.begin(), ::tolower);
                if (wordToIndex.find(word) == wordToIndex.end()) {
                    wordToIndex[word] = vocabSize;
                    indexToWord[vocabSize] = word;
                    vocabSize++;
                }
            }
        }
    }

    int encode(const std::string& word) {
        std::string lowerWord = word;
        std::transform(lowerWord.begin(), lowerWord.end(), lowerWord.begin(), ::tolower);
        if (wordToIndex.find(lowerWord) != wordToIndex.end()) {
            return wordToIndex[lowerWord];
        }
        return -1; // Unknown word
    }

    std::string decode(int index) {
        if (indexToWord.find(index) != indexToWord.end()) {
            return indexToWord[index];
        }
        return "<unk>"; // Unknown index
    }

    int getVocabSize() const {
        return vocabSize;
    }
};

// Embeddings class (Word Embeddings + Positional Encoding)
class Embeddings {
private:
    Matrix wordEmbeddings;
    Matrix positionalEncodings;
    int embeddingDim;
    int maxSequenceLength;

    void generatePositionalEncodings() {
        positionalEncodings.assign(maxSequenceLength, Vector(embeddingDim));
        for (int pos = 0; pos < maxSequenceLength; ++pos) {
            for (int i = 0; i < embeddingDim; ++i) {
                if (i % 2 == 0) {
                    positionalEncodings[pos][i] = std::sin(pos / std::pow(10000, (2.0 * i) / embeddingDim));
                } else {
                    positionalEncodings[pos][i] = std::cos(pos / std::pow(10000, (2.0 * (i - 1)) / embeddingDim));
                }
            }
        }
    }

public:
    Embeddings(int vocabSize, int embedDim, int maxSeqLen)
        : embeddingDim(embedDim), maxSequenceLength(maxSeqLen) {
        
        // Initialize word embeddings randomly
        Utils::initializeMatrix(wordEmbeddings, vocabSize, embeddingDim);

        // Generate positional encodings
        generatePositionalEncodings();
    }

    // Get embedding for a token at a specific position
    Vector getEmbedding(int tokenIndex, int position) {
        Vector embedding = wordEmbeddings[tokenIndex];
        Vector posEncoding = positionalEncodings[position];
        
        // Add word embedding and positional encoding
        return Utils::add(embedding, posEncoding);
    }

    int getEmbeddingDim() const {
        return embeddingDim;
    }

    int getMaxSequenceLength() const { // Correctly placed getter
        return maxSequenceLength;
    }
};

#endif // TOKENIZER_EMBEDDINGS_H


