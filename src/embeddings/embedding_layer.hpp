#pragma once
#include <Eigen/Dense>
#include "config.hpp"

class EmbeddingLayer {
    public:
        using EmbeddingMatrix = Eigen::Matrix<Eigen::half, cfg::vocab_size, cfg::embedding_dim>;
        // using basically says "this is the type". It's like class MyType(TypedDict) in python
        using PositionalMatrix = Eigen::Matrix<Eigen::half, cfg::max_seq_length, cfg::embedding_dim>;

        EmbeddingMatrix embeddings;
        PositionalMatrix positional_encodings;

        void build_embeddings() {}
};