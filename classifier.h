#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

// A LOT of this is taken from https://github.com/BVLC/caffe/edit/master/examples/cpp_classification/classification.cpp

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<int, int> coordinates;
typedef std::pair<coordinates, float> prediction;

class Classifier {
 public:
  Classifier(const string& model_file,
             const string& proto_file,
             const string& mean_file);


 private:
  void SetMean(const string& mean_file);

  std::vector<prediction> Predict(const cv::Mat& img);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  std::vector<string> labels_;
};
