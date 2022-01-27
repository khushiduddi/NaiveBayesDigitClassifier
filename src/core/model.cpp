//
// Created by Khushi Duddi on 4/2/21.
//

#include "core/model.h"

namespace naivebayes {
    Model::Model() {
        train_total_ = 0;
        num_pixels_ = -1;
        pixel_class_count_.resize(10);
        for (size_t s = 0; s < 10; s++) {
            pixel_class_count_[s].resize(2);
        }
        for (size_t i = 0; i < 10; i++) train_class_total_[i] = 0;
    }

    void Model::BuildModel(std::string fileName) {
        ifstream my_file;
        my_file.open(fileName);
        if (!my_file || !my_file.is_open()) {
            cout << "File open error: " << fileName << std::endl;
            return;
        }
        cout << "Building model from file: " << fileName << endl;

        while (!my_file.eof()) {
            Sample sample;
            my_file >> sample;
            if (sample.GetSampleLength() == sample.kSampleIgnore) {
                break;
            }
            ProcessSample(sample);
            if (GetSampleLength() < 0) {
                cout << "Invalid model \n";
                num_pixels_ = -1;
                return;
            }
        }
        BuildPrior();
        BuildLikelihood();

        my_file.close();
    }

    double Model::Classify(string fileName, double digit_accuracy[10]) {
        ifstream my_file;
        my_file.open(fileName);
        if (!my_file || !my_file.is_open()) {
            cout << "File open error: " << fileName << std::endl;
            return -1;
        }
        cout << "Classifying sample from file: " << fileName << endl;

        size_t passed = 0;
        size_t total = 0;
        size_t passed_digit[10] = {0};
        size_t total_digit[10] = {0};

        while (!my_file.eof()) {
            Sample sample;
            my_file >> sample;
            if (sample.GetSampleLength() == sample.kSampleIgnore) {
                break;
            }
            total++;
            total_digit[sample.GetDigit()]++;
            int classified_digit = CalculateClassification(sample);
            if (sample.GetDigit() == classified_digit) {
                passed++;
                passed_digit[classified_digit]++;
            }
        }
        double accuracy = passed * 1.0 / total;
        cout << "Accuracy of classification: " << accuracy << endl;
        for (size_t i = 0; i < kDigits; i++) {
            digit_accuracy[i] = passed_digit[i] * 1.0 / total_digit[i];
            cout << "Accuracy of " << i << ": " << digit_accuracy[i] << endl;
        }
        my_file.close();
        return accuracy;
    }

    void Model::Print() {
        if (num_pixels_ < 0) {
            // Invalid model
            return;
        }
        cout << "Total number of images: " << train_total_ << endl;
        for (int i = 0; i < 10; i++) {
            cout << "Class " << i << " prior: " << p_prior_[i] << endl;
        }

        for (int c = 0; c < 10; c++) {
            for (int v = 0; v < kNumShades; v++) {
                cout << "c: " << c << " : " << v << endl;
                for (int i = 0; i < num_pixels_; i++) {
                    for (int j = 0; j < num_pixels_; j++) {
                        //printf("%05.3f ", p_likelihood_class_pixel_[c][v][i * num_pixels_ + j]);
                        cout << p_likelihood_class_pixel_[c][v][i * num_pixels_ + j] << " ";
                    }
                    cout << endl;
                }
            }
        }
    }

    int Model::Save(string filename) {
        if (num_pixels_ < 0) {
            // Invalid model
            cout << "Could not save. Model is not valid.";
            return 0;
        }
        ofstream my_file(filename);
        if (!my_file.is_open()) {
            cout << "Cannot open file for writing: " << filename << endl;
            return 0; // error
        }
        my_file << num_pixels_ << endl;
        for (size_t i = 0; i < 10; i++) {
            my_file << p_prior_[i] << endl;
        }

        for (int c = 0; c < 10; c++) {
            for (int v = 0; v < kNumShades; v++) {
                my_file << c << " " << v << endl;
                for (int i = 0; i < num_pixels_; i++) {
                    for (int j = 0; j < num_pixels_; j++) {
                        my_file << p_likelihood_class_pixel_[c][v][i * num_pixels_ + j] << " ";
                    }
                    my_file << endl;
                }
            }
        }
        my_file.close();
        cout << "Saved model to file: " << filename << endl;
        return 1; // success
    }

    int Model::Load(string filename) {
        num_pixels_ = -1; // Invalidate model before loading a new one into it
        ifstream my_file(filename);
        if (!my_file.is_open()) {
            cout << "Cannot open file for reading: " << filename << endl;
            return 0; // error
        }
        my_file >> num_pixels_;
        for (int i = 0; i < 10; i++) {
            my_file >> p_prior_[i];
        }

        int c_file;
        int v_file;
        for (int c = 0; c < 10; c++) {
            for (int v = 0; v < kNumShades; v++) {
                p_likelihood_class_pixel_[c][v].resize(num_pixels_ * num_pixels_);
                my_file >> c_file >> v_file;
                for (int i = 0; i < num_pixels_; i++) {
                    for (int j = 0; j < num_pixels_; j++) {
                        my_file >> p_likelihood_class_pixel_[c][v][i * num_pixels_ + j];
                    }
                }
            }
        }
        my_file.close();
        cout << "Loaded model from file: " << filename << endl;
        return 1;
    }

    void Model::ProcessSample(Sample& sample) {
        if (sample.GetSampleLength() == sample.kSampleError || sample.GetSampleLength() == sample.kSampleIgnore) {
            // Sample is invalid. Do not process.
            return;
        }
        if (num_pixels_ < 0) {
            // set up dimensions after reading first sample
            num_pixels_ = sample.GetSampleLength();
            //cout << "num pixels set to: " << num_pixels_ << endl;
            for (size_t c = 0; c < 10; c++) {
                for (size_t v = 0; v < kNumShades; v++) {
                    pixel_class_count_[c][v].resize(num_pixels_ * num_pixels_);
                }
            }
        }
        if (sample.GetSampleLength() != num_pixels_) {
            cout << "Invalid sizes of images \n";
            num_pixels_ = -1;
            return;
        }
        train_total_++;
        train_class_total_[sample.GetDigit()]++;
        for (size_t i = 0; i < sample.GetImagePixels().size(); i++) {
            int val = sample.GetImagePixels()[i];
            if (sample.GetDigit() < 0 || sample.GetDigit() > 9) {
                cout << "incorrect digit: " << sample.GetDigit() << endl;
                num_pixels_ = -1; // Invalidate sample
                return;
            }

            pixel_class_count_[sample.GetDigit()][val][i]++;
        }
    }

    void Model::BuildPrior() {
        for (size_t i = 0; i < 10; i++) {
            p_prior_[i] = (kLaplace + train_class_total_[i]) / (10.0 * kLaplace + train_total_);
        }
    }

    void Model::BuildLikelihood() {
        for (size_t c = 0; c < 10; c++) {
            for (size_t v = 0; v < kNumShades; v++) {
                p_likelihood_class_pixel_[c][v].resize(num_pixels_ * num_pixels_);
                for (size_t i = 0; i < num_pixels_; i++) {
                    for (size_t j = 0; j < num_pixels_; j++) {
                        p_likelihood_class_pixel_[c][v][i * num_pixels_ + j] = (kLaplace + pixel_class_count_[c][v][i * num_pixels_ + j])
                                / (2 * kLaplace + train_class_total_[c]);
                    }
                }
            }
        }
    }

    int Model::GetSampleTotals() {
        return train_total_;
    }

    double Model::GetLikelihood(int digit, int value, int row, int column) {
        if (row < 0 || row >= num_pixels_ ||
            column < 0 || column >= num_pixels_ ||
            value < 0 || value >= kNumShades ||
            digit < 0 || digit > 9 || num_pixels_ < 0) {
            return -1.0;
        }
        return p_likelihood_class_pixel_[digit][value][row * num_pixels_ + column];
    }

    double Model::GetPrior(int digit) {
        if (digit < 0 || digit > 9 || num_pixels_ < 0) {
            return -1;
        }
        return p_prior_[digit];
    }

    int Model::GetSampleLength() {
        return num_pixels_;
    }

    int Model::CalculateClassification(Sample &sample) {
        if (sample.GetSampleLength() != num_pixels_) {
            cout << "Invalid sample dimensions." << endl;
            return -1;
        }
        // Computing
        double p_bayes[10];
        for (int i = 0; i < kDigits; i++) {
            p_bayes[i] = p_prior_[i];
            for (int r = 0; r < num_pixels_; r++) {
                for (int c = 0; c < num_pixels_; c++) {
                    int v = sample.GetImagePixels()[r * num_pixels_ + c];
                    p_bayes[i] *= p_likelihood_class_pixel_[i][v][r * num_pixels_ + c];
                }
            }
        }

        // Comparing
        int bayes_digit = p_bayes[0];
        for (int j = 0; j < kDigits; j++) {
            if (p_bayes[j] > p_bayes[bayes_digit]) {
                bayes_digit = j;
            }
        }
        return bayes_digit;
    }
}

// trying to read data
// how do i pass filename as argument to program