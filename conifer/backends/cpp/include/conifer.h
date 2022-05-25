#ifndef CONIFER_CPP_H__
#define CONIFER_CPP_H__
#include "json.hpp"
#include <cassert>
#include <fstream>

namespace conifer{

template<class T, class U>
class DecisionTree{

private:
  std::vector<int> feature;
  std::vector<int> children_left;
  std::vector<int> children_right;
  std::vector<T> threshold_;
  std::vector<U> value_;
  std::vector<double> threshold;
  std::vector<double> value;

public:

  U decision_function(std::vector<T> x) const{
    /* Do the prediction */
    int i = 0;
    while(feature[i] != -2){ // continue until reaching leaf
      bool comparison = x[feature[i]] <= threshold_[i];
      i = comparison ? children_left[i] : children_right[i];
    }
    return value_[i];
  }

  void init_(){
    /* Since T, U types may not be readable from the JSON, read them to double and the cast them here */
    std::transform(threshold.begin(), threshold.end(), std::back_inserter(threshold_),
                   [](double t) -> T { return (T) t; });
    std::transform(value.begin(), value.end(), std::back_inserter(value_),
                   [](double v) -> U { return (U) v; });
  }

  // Define how to read this class to/from JSON
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(DecisionTree, feature, children_left, children_right, threshold, value);

}; // class DecisionTree

template<class T, class U>
class BDT{

private:

  int n_classes;
  int n_trees;
  int n_features;
  std::vector<double> init_predict;
  std::vector<U> init_predict_;
  // vector of decision trees: outer dimension tree, inner dimension class
  std::vector<std::vector<DecisionTree<T,U>>> trees;

public:

  // Define how to read this class to/from JSON
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(BDT, n_classes, n_trees, n_features, init_predict, trees);

  BDT(std::string filename){
    /* Construct the BDT from conifer cpp backend JSON file */
    std::ifstream ifs(filename);
    nlohmann::json j = nlohmann::json::parse(ifs);
    from_json(j, *this);
    /* Do some transformation to initialise things into the proper emulation T, U types */
    if(n_classes == 2) n_classes = 1;
    std::transform(init_predict.begin(), init_predict.end(), std::back_inserter(init_predict_),
                   [](double ip) -> U { return (U) ip; });
    for(int i = 0; i < n_trees; i++){
      for(int j = 0; j < n_classes; j++){
        trees.at(i).at(j).init_();
      }
    }
  }

  std::vector<U> decision_function(std::vector<T> x) const{
    /* Do the prediction */
    assert("Size of feature vector mismatches expected n_features" && x.size() == n_features);
    std::vector<U> values;
    values.resize(n_classes, U(0));
    for(int i = 0; i < n_classes; i++){
      values.at(i) = init_predict_.at(i);
    }
    for(int i = 0; i < n_trees; i++){
      for(int j = 0; j < n_classes; j++){
        values.at(j) += trees.at(i).at(j).decision_function(x);
      }
    }
    return values;
  }

  std::vector<double> _decision_function_double(std::vector<double> x) const{
    /* Do the prediction with data in/out as double, cast to T, U before prediction */
    std::vector<T> xt;
    std::transform(x.begin(), x.end(), std::back_inserter(xt),
                   [](double xi) -> T { return (T) xi; });
    std::vector<U> y = decision_function(xt);
    std::vector<double> yd;
    std::transform(y.begin(), y.end(), std::back_inserter(yd),
                [](U yi) -> double { return (double) yi; });
    return yd;
  }

}; // class BDT

} // namespace conifer

#endif