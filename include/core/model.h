//
// Created by Khushi Duddi on 4/2/21.
//

#ifndef NAIVE_BAYES_MODEL_H
#define NAIVE_BAYES_MODEL_H

#include <iostream>
#include <fstream>
#include <vector>
#include "core/sample.h"

using std::ifstream;
using std::ofstream;

namespace naivebayes {
    class Model {
    public:
        /**
         * Constructor
         */
        Model();

        /**
         * This method builds the model from a file.
         * @param fileName
         */
        void BuildModel(std::string fileName);

        /**
         * This method prints the model.
         */
        void Print();

        /**
         * This method saves a trained model to a file.
         * @param filename
         * @return int for error checking
         */
        int Save(string filename);

        /**
         * This method loads a file back into a model.
         * @param filename
         * @return int for error checking
         */
        int Load(string filename);

        /**
         * This method calculates the total number of samples in a model.
         * @return sample total
         */
        int GetSampleTotals() ;

        /**
         * This method returns the likelihood of a pixel being shaded or unshaded.
         * @param digit
         * @param value
         * @param row
         * @param column
         * @return double probability
         */
        double GetLikelihood(int digit, int value, int row, int column);

        /**
         * This method calculates the prior of a digit in the model.
         * @param digit
         * @return double
         */
        double GetPrior(int digit);

        /**w
         * This method returns the dimension of pixels in each sample in the model.
         * @return int
         */
        int GetSampleLength();
        void ProcessSample(Sample& sample);

        double Classify(string filename, double digit_accuracy[10]);
        int CalculateClassification(Sample& sample);

    private:
        int train_class_total_[10];
        int train_total_;
        // 10 x 2 x num_pixels_^2
        vector<vector<vector<int>>> pixel_class_count_;
        int num_pixels_;
        // values for pixels can be 0..1
        const int kNumShades = 2;
        const double kLaplace = 1.0;
        const int kDigits = 10;
        // Probabilities
        double p_prior_[10];
        vector<double> p_likelihood_class_pixel_[10][2];



        void BuildPrior();
        void BuildLikelihood();

    };
}


#endif //NAIVE_BAYES_MODEL_H
