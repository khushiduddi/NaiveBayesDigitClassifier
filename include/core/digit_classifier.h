#include <string>
#include "model.h"

namespace naivebayes {

class DigitClassifier {
 public:
    DigitClassifier();

    void BuildModel(std::string fileName);
    int LoadModel(std::string fileName);
    int SaveModel(std::string fileName);
    void PrintModel();
    bool CheckSampleTotals();

private:
    Model model_;
};

}  // namespace naivebayes
