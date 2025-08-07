#ifndef TRANSFORMER_TYPES_H
#define TRANSFORMER_TYPES_H

#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

// Definiciones de tipos para mayor claridad
typedef std::vector<float> Vector;
typedef std::vector<std::vector<float>> Matrix;

// Funciones de utilidad para operaciones matriciales y vectoriales
namespace Utils {

    // Producto punto de dos vectores
    float dotProduct(const Vector& a, const Vector& b) {
        float result = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            result += a[i] * b[i];
        }
        return result;
    }

    // Multiplicación de vector por matriz (vector * matrix)
    Vector matMul(const Vector& vec, const Matrix& matrix) {
        Vector result(matrix[0].size(), 0.0);
        for (size_t i = 0; i < matrix[0].size(); ++i) {
            for (size_t j = 0; j < vec.size(); ++j) {
                result[i] += vec[j] * matrix[j][i];
            }
        }
        return result;
    }

    // Multiplicación de matriz por vector (matrix * vector)
    Vector matMul(const Matrix& matrix, const Vector& vec) {
        Vector result(matrix.size(), 0.0);
        for (size_t i = 0; i < matrix.size(); ++i) {
            for (size_t j = 0; j < vec.size(); ++j) {
                result[i] += matrix[i][j] * vec[j];
            }
        }
        return result;
    }

    // Suma de dos vectores
    Vector add(const Vector& a, const Vector& b) {
        Vector result(a.size());
        for (size_t i = 0; i < a.size(); ++i) {
            result[i] = a[i] + b[i];
        }
        return result;
    }

    // Softmax
    Vector softmax(const Vector& scores) {
        Vector expScores;
        float maxScore = *std::max_element(scores.begin(), scores.end());
        for (float score : scores) {
            expScores.push_back(std::exp(score - maxScore));
        }
        float sumExpScores = std::accumulate(expScores.begin(), expScores.end(), 0.0f);
        Vector result(scores.size());
        for (size_t i = 0; i < scores.size(); ++i) {
            result[i] = expScores[i] / sumExpScores;
        }
        return result;
    }

    // Inicialización de matriz con valores aleatorios
    void initializeMatrix(Matrix& matrix, int rows, int cols) {
        matrix.assign(rows, Vector(cols));
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                matrix[i][j] = (float)rand() / RAND_MAX - 0.5; // Valores entre -0.5 y 0.5
            }
        }
    }

    // Normalización de capa (Layer Normalization)
    Vector layerNorm(const Vector& input, const Vector& gamma, const Vector& beta, float epsilon = 1e-5) {
        float mean = std::accumulate(input.begin(), input.end(), 0.0f) / input.size();
        float variance = 0.0f;
        for (float val : input) {
            variance += (val - mean) * (val - mean);
        }
        variance /= input.size();

        Vector output(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = gamma[i] * (input[i] - mean) / std::sqrt(variance + epsilon) + beta[i];
        }
        return output;
    }

}

#endif // TRANSFORMER_TYPES_H


