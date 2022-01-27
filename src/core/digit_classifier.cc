#include <core/digit_classifier.h>
#include <iostream>
#include <fstream>

namespace naivebayes {

    DigitClassifier::DigitClassifier() {}

    void DigitClassifier::BuildModel(std::string fileName) {
        model_.BuildModel(fileName);
    }

    bool DigitClassifier::CheckSampleTotals() {
        return model_.GetSampleTotals();
    }

    int DigitClassifier::LoadModel(std::string fileName) {
        cout << "Digit_classifier loading model from file: " << fileName << endl;
        return model_.Load(fileName);
    }

    int DigitClassifier::SaveModel(std::string fileName) {
        cout << "Digit_classifier saving model from file: " << fileName << endl;
        return model_.Save(fileName);
    }

    void DigitClassifier::PrintModel() {
        model_.Print();
    }

}  // namespace naivebayes