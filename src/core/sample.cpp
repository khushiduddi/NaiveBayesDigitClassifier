//
// Created by Khushi Duddi on 4/3/21.
//

#include "core/sample.h"

namespace naivebayes {
    Sample::Sample(int numPixel): num_pixels_(numPixel) {
        // digit will change after input is read
        digit_ = -1;
        if (numPixel > 0) {
            image_pixels_.resize(numPixel * numPixel);
            std::fill(image_pixels_.begin(), image_pixels_.end(), 0);
        }
    }

    Sample::Sample(string fileName) {
        ifstream my_file;
        my_file.open(fileName);
        if (!my_file || !my_file.is_open()) {
            cout << "File open error: " << fileName << std::endl;
            return;
        }
        my_file >> *this;
        my_file.close();
    }

    istream &operator>>(istream &input, Sample &sample) {
        string line;
        if (!getline(input, line)) {
            // Last line of file, ignore sample
            sample.num_pixels_ = sample.kSampleIgnore;
            return input;
        }
        if (line.length() != 1) {
            cout << "Training data format error, expected digit line length issue: " << line << endl;
            return input;
        }
        sample.digit_ = stoi(line);
        if (sample.digit_ < 0 || sample.digit_ > 9) {
            cout << "Training data format error, expected digit: " << line << endl;
            return input;
        }
        int n = 0;
        while (n < sample.num_pixels_ || sample.num_pixels_ < 0) {
            if (!getline(input, line)) {
                cout << "Training data format error: " << line << endl;
                return input;
            }
            if (n == 0) {
                sample.num_pixels_ = line.length();
                sample.image_pixels_.resize(sample.num_pixels_ * sample.num_pixels_); // first lines length = image dimension
            } else {
                if (line.length() != sample.GetSampleLength()) {
                    cout << "Lines are not the same length. Invalid";
                    sample.num_pixels_ = sample.kSampleError;
                    return input;
                }
            }
            // Process the line
            for (size_t i = 0; i < line.length(); i++) {
                int shade = 1;
                if (line[i] == ' ') {
                    shade = 0;
                }
                sample.image_pixels_[n * sample.num_pixels_ + i] = shade;
            }
            n++;
        }
        return input;
    }

    int Sample::GetDigit() {
        return digit_;
    }

    int Sample::GetSampleLength() {
        return num_pixels_;
    }

    vector<int> &Sample::GetImagePixels() {
        return image_pixels_;
    }

    int Sample::GetPixel(size_t row, size_t col) const {
        if (row >= num_pixels_ || col >= num_pixels_) {
            return -1;
        }
        return image_pixels_[row * num_pixels_ + col];
    }

    int Sample::SetPixel(size_t row, size_t col, size_t shade) {
        if (row >= num_pixels_ || col >= num_pixels_ || shade >= 2) {
            return -1;
        }
        image_pixels_[row * num_pixels_ + col] = shade;
        return 0;
    }

    void Sample::Clear() {
        std::fill(image_pixels_.begin(), image_pixels_.end(), 0);
    }
}