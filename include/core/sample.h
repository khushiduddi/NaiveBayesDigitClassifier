//
// Created by Khushi Duddi on 4/3/21.
//

#ifndef NAIVE_BAYES_SAMPLE_H
#define NAIVE_BAYES_SAMPLE_H

#include <iostream>
#include <vector>
#include <fstream>

using std::ifstream;
using std::vector;
using std::string;
using std::istream;
using std::cout;
using std::endl;

namespace naivebayes {
    class Sample {
    public:
        Sample(int numPixel = -1);
        Sample(string fileName);

        friend istream& operator>>(istream& input, Sample& sample);

        int GetDigit();
        int GetSampleLength();
        int SetPixel(size_t row, size_t col, size_t shade);
        int GetPixel(size_t row, size_t col) const;
        vector<int> &GetImagePixels();
        void Clear();

        // For error checking of return values of GetSampleLength()
        const int kSampleIgnore = -2;
        const int kSampleError = -1;

    private:
        size_t digit_;
        size_t num_pixels_;
        vector<int> image_pixels_;
    };
}

#endif //NAIVE_BAYES_SAMPLE_H
