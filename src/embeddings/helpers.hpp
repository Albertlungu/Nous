#pragma once
#include <iostream>
#include <Eigen/Dense>
#include "config.hpp"

namespace helpers {
    Eigen::Matrix<int, batch.size(), cfg::max_seq_length> pad_token_ids(
        int batch_size,
        std::vector<std::vector<int>> batch,
        int pad_token_id = 0
    ) {
        Eigen::Matrix<int, cfg::batch_size, cfg::max_seq_length> eigen_batches;

        for (int seq = 0; seq < cfg::batch_size; seq++) {
            if (batch[seq].size() < cfg::max_seq_length) {
                batch[seq].insert(batch[seq].end(), cfg::max_seq_length - batch[seq].size(), pad_token_id);
            } else if (batch[seq].size() > cfg::max_seq_length) {
                batch[seq].resize(cfg::max_seq_length - batch[seq].size());
            }

            eigen_batches.row(seq) = Eigen::Map<Eigen::Matrix<int, 1, cfg::max_seq_length>>(batch[seq].data());

        }
    }
}