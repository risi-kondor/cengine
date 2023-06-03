#include "Cengine_base.cpp"
#include "CtensorObject.hpp"

using namespace Cengine;

typedef CtensorObject Ctensor;

int main(int argc, char** argv) {
  Ctensor A({4, 4}, fill::gaussian);
  Ctensor B({4, 4}, fill::gaussian);

  cout << endl << "A=" << endl << A << endl << endl;
  cout << endl << "B=" << endl << B << endl << endl;

  cout << endl << "A+A=" << endl << A + A << endl << endl;

  A += B;
  A += B;
}
