//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "firmware/my_prj.h"
#include "firmware/parameters.h"
#include "firmware/BDT.h"

#define CHECKPOINT 5000

int main(int argc, char **argv)
{
  //load input data from text file
  std::ifstream fin("tb_data/tb_input_features.dat");

#ifdef RTL_SIM
  std::string RESULTS_LOG = "tb_data/cosim_results.log";
  std::string VERBOSE_LOG = "tb_data/cosim_tree_results.log";
#else
  std::string RESULTS_LOG = "tb_data/csim_results.log";
  std::string VERBOSE_LOG = "tb_data/csim_tree_results.log";
#endif
  std::ofstream fout(RESULTS_LOG);
  std::ofstream ftrees(VERBOSE_LOG);

  std::string iline;
  std::string pline;
  int e = 0;

  if (fin.is_open()) {
    while ( std::getline(fin,iline) ) {
      if (e % CHECKPOINT == 0) std::cout << "Processing input " << e << std::endl;
      e++;
      char* cstr=const_cast<char*>(iline.c_str());
      char* current;
      std::vector<float> in;
      current=strtok(cstr," ");
      while(current!=NULL) {
        in.push_back(atof(current));
        current=strtok(NULL," ");
      }

      //hls-fpga-machine-learning insert data
      std::vector<float>::const_iterator in_begin = in.cbegin();
      std::vector<float>::const_iterator in_end;
      input_arr_t x;
      in_end = in_begin + (10);
      std::copy(in_begin, in_end, x);
      in_begin = in_end;
      score_arr_t score{};
      score_t tree_scores[BDT::fn_classes(n_classes) * n_trees]{};
      std::fill_n(score, 2, 0.);

      //hls-fpga-machine-learning insert top-level-function
      my_prj(x, score, tree_scores);

      for(int i = 0; i < BDT::fn_classes(n_classes); i++){
        fout << score[i] << " ";
      }
      fout << std::endl;
      for(int  i = 0; i < n_trees; i++){
          for(int j = 0; j < BDT::fn_classes(n_classes); j++){
            ftrees << tree_scores[i * BDT::fn_classes(n_classes) + j] << " ";
          }
      }
      ftrees << std::endl;

      if (e % CHECKPOINT == 0) {
        std::cout << "Quantized predictions" << std::endl;
        //hls-fpga-machine-learning insert quantized
        for(int i = 0; i < 2; i++) {
          std::cout << score[i] << " ";
        }
        std::cout << std::endl;
      }
    }
    fin.close();
  } else {
    std::cout << "INFO: Unable to open input file, using default input." << std::endl;
    //hls-fpga-machine-learning insert zero
    input_arr_t x;
    std::fill_n(x, 10, 0.);
    score_arr_t score{};
    score_t tree_scores[BDT::fn_classes(n_classes) * n_trees]{};
    std::fill_n(score, 2, 0.);

    //hls-fpga-machine-learning insert top-level-function
    my_prj(x, score, tree_scores);

    //hls-fpga-machine-learning insert output
    for(int i = 0; i < 2; i++) {
      std::cout << score[i] << " ";
    }
    std::cout << std::endl;

    //hls-fpga-machine-learning insert tb-output
    for(int i = 0; i < 2; i++) {
      fout << score[i] << " ";
    }
  }

  fout.close();
  ftrees.close();
  std::cout << "INFO: Saved inference results to file: " << RESULTS_LOG << std::endl;

  return 0;
}
