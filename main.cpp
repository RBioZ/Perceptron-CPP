#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <vector>

using namespace std;
using matrix = vector<std::vector<float>>;

class Matriz:vector<vector<float>>{
  public:
    Matriz(){

    };
};

matrix map (matrix& m1,float (*function)(float)){

    matrix result(m1.size(), std::vector<float>(m1.at(0).size()));

    for(std::size_t row = 0; row < result.size(); ++row) {
      for(std::size_t col = 0; col < result.at(0).size(); ++col) {
        result[row][col] = function(m1[row][col]);
      }
    }

    return result;
}

float sigmoid(float x){
  return 1 / (1 + exp(-x));
}

float dsigmoid(float x){
  return x * (1 - x);
}

void print(const matrix& m)
{
    for(const auto& v : m) {
        for(const float i : v) { std::cout << i << ' '; }
        std::cout << '\n';
    }
    std::cout << '\n';
}

matrix transpose(const matrix& m)
{
    matrix outtrans(m.at(0).size(), std::vector<float>(m.size()));
    for (std::size_t i = 0; i < m.size(); ++i) {
        for (std::size_t j = 0; j < m.at(0).size(); ++j) {
            outtrans.at(j).at(i) = m.at(i).at(j);
        }
    }
    return outtrans;
}

matrix multiply(const matrix& m1, const matrix& m2)
{
    matrix result(m1.size(), std::vector<float>(m2.at(0).size()));

    for(std::size_t row = 0; row < result.size(); ++row) {
        for(std::size_t col = 0; col < result.at(0).size(); ++col) {
            for(std::size_t inner = 0; inner < m2.size(); ++inner) {
                result.at(row).at(col) += m1.at(row).at(inner) * m2.at(inner).at(col);
            }
        }
    }
    return result;
}

matrix sum(const matrix& m1, const matrix& m2)
{
    matrix result(m1.size(), std::vector<float>(m2.at(0).size()));

    for(std::size_t row = 0; row < result.size(); ++row) {
        for(std::size_t col = 0; col < result.at(0).size(); ++col) {
          result[row][col] = m1[row][col] + m2[row][col];
        }
    }
    return result;
}

matrix subtract(const matrix& m1, const matrix& m2)
{
    matrix result(m1.size(), std::vector<float>(m2.at(0).size()));

    for(std::size_t row = 0; row < result.size(); ++row) {
        for(std::size_t col = 0; col < result.at(0).size(); ++col) {
          result[row][col] = m1[row][col] - m2[row][col];
        }
    }
    return result;
}

matrix escalar_multiply(const matrix& m1, float escalar)
{
    matrix result(m1.size(), std::vector<float>(m1.at(0).size()));

    for(std::size_t row = 0; row < result.size(); ++row) {
        for(std::size_t col = 0; col < result.at(0).size(); ++col) {
          result[row][col] = m1[row][col] * escalar;
        }
    }
    return result;
}

matrix hadamard(const matrix& m1, const matrix& m2)
{
    matrix result(m1.size(), std::vector<float>(m2.at(0).size()));

    for(std::size_t row = 0; row < result.size(); ++row) {
        for(std::size_t col = 0; col < result.at(0).size(); ++col) {
          result[row][col] = m1[row][col] * m2[row][col];
        }
    }
    return result;
}

matrix randomize(int m, int n){
  matrix result(m, std::vector<float>(n));

  for(std::size_t row = 0; row < result.size(); ++row) {
    for(std::size_t col = 0; col < result.at(0).size(); ++col) {
      result[row][col] = (float) rand()/RAND_MAX;
    }
  }

  return result;
}

class Perceptron {
  public:

    int i_nodes, h_nodes, o_nodes;
    float learning_rate = 0.1;
    matrix weights_ih, weights_ho;
    matrix bias_ih, bias_ho;


    Perceptron(int i_nodes, int h_nodes, int o_nodes){
      this->i_nodes = i_nodes;
      this->h_nodes = h_nodes;
      this->o_nodes = o_nodes;

      weights_ih = randomize(h_nodes, i_nodes);
      weights_ho = randomize(o_nodes, h_nodes);
      bias_ih = randomize(h_nodes, 1);
      bias_ho = randomize(o_nodes, 1);

    }

    void train(matrix input, matrix target){


      // <-----FEEDFOWARD----->

      // Input -> Hidden
      matrix hidden = multiply(weights_ih, input);
      hidden = sum(hidden, bias_ih);
      hidden = map(hidden, sigmoid);
      
      // Hidden -> Output
      matrix output = multiply(weights_ho, hidden);
      output = sum(output, bias_ho);
      output = map(output, sigmoid);

      // <-----BACKPROPAGATION---->

      // Output -> Hidden
      matrix output_error = subtract(target, output);
      matrix d_output = map(output,dsigmoid);
      matrix hiddent_T = transpose(hidden);
      matrix gradient = hadamard(d_output, output_error);
      gradient = escalar_multiply(gradient, this->learning_rate);

      // Adjust Bias O -> H
      bias_ho = sum(bias_ho, gradient);
      // Adjust Weights O -> H
      matrix weights_ho_deltas = multiply(gradient, hiddent_T);
      weights_ho = sum(weights_ho, weights_ho_deltas);

      // Hidden -> Input
      matrix weights_ho_T = transpose(weights_ho);
      matrix hidden_error = multiply(weights_ho_T, output_error);
      matrix d_hidden = map(hidden, dsigmoid);
      matrix input_T = transpose(input);

      matrix gradient_H = hadamard(d_hidden, hidden_error);
      gradient_H = escalar_multiply(gradient_H, this->learning_rate);

      // Adjust Bias H -> I
      bias_ih = sum(bias_ih, gradient_H);
      // Adjust Weights H -> I
      matrix weights_ih_deltas = multiply(gradient_H, input_T);
      weights_ih = sum(weights_ih, weights_ih_deltas);
    }

    void predict(matrix input){

      // Input -> Hidden
      matrix hidden = multiply(weights_ih, input);
      hidden = sum(hidden, bias_ih);
      hidden = map(hidden, sigmoid);
      
      // Hidden -> Output
      matrix output = multiply(weights_ho, hidden);
      output = sum(output, bias_ho);
      output = map(output, sigmoid);

      print(output);
    }
};

int main(){

  Perceptron NeuralNetwork(2,3,1);
  matrix input {
    {0},
    {1}
  };

  matrix output {
    {0},
  };

  for(int i = 0; i < 10000; i++){
    NeuralNetwork.train(input, output);
  }

  NeuralNetwork.predict(input);

  return 0;
}