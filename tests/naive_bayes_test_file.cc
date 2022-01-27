#include <catch2/catch.hpp>

#include "core/digit_classifier.h"
#include "core/model.h"
#define TWO_DECIMALS(x) (round(x * 100)/100)

TEST_CASE("Check consistency of reading training data from file, making sure the total equals the sum of all class samples") {
    SECTION("Checking sample total of test file") {
        naivebayes::Model model;
        model.BuildModel("../../../../../../tests/testimages.txt");
        REQUIRE(model.GetSampleTotals() == 5);
    }

    SECTION("BuildModel with a nonexistent file") {
        naivebayes::Model model;
        model.BuildModel("../../../../../../tests/testdoesnotexist.txt");
        REQUIRE(model.GetSampleLength() == -1);
    }

}
TEST_CASE("Check if the prior calculated is correct based off of the equation.") {
    naivebayes::Model model;
    model.BuildModel("../../../../../../tests/testimages.txt");

    SECTION("Checking prior of class 3 in test file.") {
        /*
         * k = 1
         * prior of 3 = (k + 1)/(10k + 5) = 0.1333
         */
        REQUIRE(TWO_DECIMALS(model.GetPrior(3)) == 0.13);
    }

    SECTION("Checking prior of image that does not exist in training data.") {
        /*
         * k = 1
         * prior of 9 = (k + 0)/(10k + 5) = 0.067
         * 9 is not in training data
         */
        REQUIRE(TWO_DECIMALS(model.GetPrior(9)) == 0.07);
    }

    SECTION("Checking prior of invalid digit") {
        REQUIRE(TWO_DECIMALS(model.GetPrior(10)) == -1);
    }
}

TEST_CASE("Check if the likelihood of each class is correct.") {
    naivebayes::Model model;
    model.BuildModel("../../../../../../tests/testimages.txt");
    model.Print();

    SECTION("Checking likelihood of digit 3 value 0, place 1,1") {
        double p = model.GetLikelihood(3, 0, 1, 1);
        /*
         * k = 1
         * likelihood = (k + 0)/(2k + 1) = 1/3 = 0.333
         */
        REQUIRE(TWO_DECIMALS(p) == 0.33);
    }

    SECTION("Checking likelihood of digit 3 value 1, place 1,1") {
        double p = model.GetLikelihood(3, 1, 1, 1);
        /*
         * k = 1
         * likelihood = (k + 0)/(2k + 1) = 2/3 = 0.667
         */
        REQUIRE(TWO_DECIMALS(p) == 0.67);
    }

    SECTION("Checking likelihood of class that does not exist in training data") {
        /*
         * k = 1
         * likelihood = (k + 0)/(2k + 0) = 1/5 = 0.5
         * 9 does not exist in training data
         */
        REQUIRE(TWO_DECIMALS(model.GetLikelihood(9, 0, 1, 1)) == 0.5);
    }

    SECTION("Checking likelihood of invalid digit") {
        REQUIRE(TWO_DECIMALS(model.GetLikelihood(10, 1, 0, 0)) == -1);
    }
}

TEST_CASE("Testing Save and Load methods") {
    naivebayes::Model model1;
    naivebayes::Model model2;
    model1.BuildModel("../../../../../../tests/testimages.txt");

    SECTION("Checking saving model to a file making sure the file exists.") {
        model1.Save("test.txt");
        REQUIRE(std::__fs::filesystem::exists("test.txt"));
    }

    SECTION("Checking loading back a model.") {
        model2.Load("test.txt");
        // Spot checking to see if models are the same
        REQUIRE(TWO_DECIMALS(model1.GetPrior(2)) == TWO_DECIMALS(model2.GetPrior(2)));
    }

    SECTION("Checking loading of nonexistent file.") {
        model2.Load("doesnotexist.txt");
        REQUIRE(!std::__fs::filesystem::exists("doesnotexist.txt"));
        REQUIRE(model2.GetSampleLength() == -1);
    }
    SECTION("Checking saving to non writable file") {
        model1.Save("/doesnotexist/test.txt");
        REQUIRE(!std::__fs::filesystem::exists("/doesnotexist/test.txt"));
    }
}

TEST_CASE("Building models for files that are not formatted correctly.") {
    SECTION("Testing BuildModel for a file where the samples have different dimensions") {
        naivebayes::Model model;
        model.BuildModel("../../../../../../tests/testinvalidimages.txt");
        REQUIRE(model.GetSampleLength() == -1);
    }
    SECTION("Testing BuildModel for a file where the samples do not have square dimensions and have different line lengths.") {
        naivebayes::Model model;
        model.BuildModel("../../../../../../tests/testinvalidlinelengths.txt");
        REQUIRE(model.GetSampleLength() == -1);
    }
}

TEST_CASE("Checking to see if the training data file with large number of training samples builds properly.") {
    naivebayes::Model model;
    model.BuildModel("../../../../../../tests/trainingimagesandlabels.txt");
    REQUIRE(model.GetSampleTotals() == 5000);
}

TEST_CASE("Tests to see if samples are correctly classified.") {
    naivebayes::Model model;
    model.BuildModel("../../../../../../tests/trainingimagesandlabels.txt");
}

TEST_CASE("Testing accuracy of classification for each digit in known test data.") {
    naivebayes::Model model;
    model.BuildModel("../../../../../../tests/trainingimagesandlabels.txt");
    double digit_accuracy[10] = {0};
    double accuracy = model.Classify("../../../../../../tests/testimagesandlabels.txt", digit_accuracy);
    cout << "Test accuracy: " << accuracy << endl;
    REQUIRE(accuracy > 0.70);
    for (size_t i = 0; i < 10; i++) {
        REQUIRE(digit_accuracy[i] > 0.6);
    }
}

TEST_CASE("Testing classification of file with one sample") {
    naivebayes::Sample sample("../../../../../../tests/testoneimage.txt");
    naivebayes::Model model;
    model.BuildModel("../../../../../../tests/trainingimagesandlabels.txt");
    REQUIRE(model.CalculateClassification(sample) == 5);
}

TEST_CASE("Test classification of different types of invalid samples.") {
    SECTION("Test incorrect sample size") {
        naivebayes::Sample sample("../../../../../../tests/testinvalidimages.txt");
        naivebayes::Model model;
        model.BuildModel("../../../../../../tests/trainingimagesandlabels.txt");
        REQUIRE(model.CalculateClassification(sample) == -1);
    }
    SECTION("Test invalid format of sample.") {
        naivebayes::Sample sample("../../../../../../tests/testinvalidlinelengths.txt");
        naivebayes::Model model;
        model.BuildModel("../../../../../../tests/trainingimagesandlabels.txt");
        REQUIRE(model.CalculateClassification(sample) == -1);
    }
    SECTION("Test classification of a non-existent sample file") {
        naivebayes::Sample sample("../../../../../../tests/doesnotexist.txt");
        naivebayes::Model model;
        model.BuildModel("../../../../../../tests/trainingimagesandlabels.txt");
        REQUIRE(model.CalculateClassification(sample) == -1);
    }
}

TEST_CASE("Test public methods in sample: getters and setters") {
    SECTION("Test method in sample class GetSampleLength") {
        naivebayes::Sample sample("../../../../../../tests/testoneimage.txt");
        REQUIRE(sample.GetSampleLength() == 28);
    }
    SECTION("Test method in sample class GetPixel and SetPixel") {
        naivebayes::Sample sample("../../../../../../tests/testoneimage.txt");
        REQUIRE(sample.GetPixel(0, 0) == 0);
        sample.SetPixel(0, 0, 1);
        REQUIRE(sample.GetPixel(0, 0) == 1);
    }
    SECTION("Test method in sample class GetPixel with invalid parameter.") {
        naivebayes::Sample sample("../../../../../../tests/testoneimage.txt");
        REQUIRE(sample.GetPixel(-10, -2) == -1);
        REQUIRE(sample.GetPixel(100, 200) == -1);
    }
    SECTION("Test method in sample class SetPixel with invalid parameter.") {
        naivebayes::Sample sample("../../../../../../tests/testoneimage.txt");
        REQUIRE(sample.SetPixel(100, 200, 10) == -1);
    }
}
