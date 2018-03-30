#pragma once

#include <string>

#include "cereal/cereal.hpp"
#include "Eigen/Dense"

namespace cereal {
    template <class Archive, class _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols> inline
    void
    save(Archive & ar, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> const & m) {
        int32_t rows = m.rows();
        int32_t cols = m.cols();
        ar(cereal::make_nvp("rows", rows));
        ar(cereal::make_nvp("cols", cols));
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                ar(cereal::make_nvp("(" + std::to_string(i) + ", " + std::to_string(j) + ")",
                                    m(i, j)));
            }
        }
    }
    
    template <class Archive, class _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols> inline
    void load(Archive & ar, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> & m) {
        int32_t rows;
        int32_t cols;
        ar(rows);
        ar(cols);

        m.resize(rows, cols);
        
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                ar(m(i, j));
        
    }
}
