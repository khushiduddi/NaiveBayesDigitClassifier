#include <iostream>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <core/digit_classifier.h>
namespace options = boost::program_options;

// Forward declaration of local helper function
int ProcessArguments(int argc, char* argv[], string& trainFile, string& saveFile, string& loadFile, string& classifyFile, int& printModel);

int main(int argc, char* argv[]) {
    string trainFile;
    string saveFile;
    string loadFile;
    string classifyFile;
    int printModel = 0;
    ProcessArguments(argc, argv, trainFile, saveFile, loadFile, classifyFile, printModel);
    naivebayes::Model model;
    if (trainFile != "") {
        model.BuildModel(trainFile);
    }
    if (saveFile != "") {
        model.Save(saveFile);
    }
    if (loadFile != "") {
        model.Load(loadFile);
    }
    if (classifyFile != "") {
        double digit_accuracy[10] = {0};
        model.Classify(classifyFile, digit_accuracy);
    }
    if (printModel != 0) {
        model.Print();
    }
}

int ProcessArguments(int argc, char* argv[], string& trainFile, string& saveFile, string& loadFile, string& classifyFile, int& printModel) {
    // Booster command line processing
    // Declare the supported options.
    options::options_description desc("Allowed options");
    desc.add_options()
            ("help", "produce help message")
            ("train", options::value<string>(), "Training data file to train model")
            ("save", options::value<string>(), "Save model to file")
            ("load", options::value<string>(), "Load model from file")
            ("classify", options::value<string>(), "Classify samples in file")
            ("print", "Print model")
            ;

    options::variables_map vm;
    options::store(options::parse_command_line(argc, argv, desc), vm);
    options::notify(vm);

    if (vm.count("help")) {
        cout << desc << endl;
        return 0;
    }
    if (vm.count("train")) {
        trainFile = vm["train"].as<string>();
    }
    if (vm.count("save")) {
        saveFile = vm["save"].as<string>();
    }
    if (vm.count("load")) {
        loadFile = vm["load"].as<string>();
    }
    if (vm.count("classify")) {
        classifyFile = vm["classify"].as<string>();
    }
    printModel = 0;
    if (vm.count("print")) {
        printModel = 1;
    }
    return 0;
}
