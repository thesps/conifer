#include "graph.hpp"

theGraph g; // instantiate the ADF graph

int main() {
  g.init();
  g.run(iter_count);
  g.end();
  return 0;
}