#ifndef _CengineTraceback
#define _CengineTraceback

#include "Cengine_base.hpp"

namespace Cengine {

class CengineTraceback {
 public:
  int nmsg = 20;

  vector<string> msg;
  int head = 0;

 public:
  CengineTraceback() { msg.resize(nmsg); }

  void operator()(const string s) {
    msg[head] = s;
    head = (head + 1) % nmsg;
  }

  void dump() {
    CoutLock lk;
    cout << endl;
    cout << "\e[1mCengine traceback:\e[0m" << endl << endl;
    for (int i = 0; i < nmsg; i++) {
      if (msg[(head + i) % nmsg] != "") {
        cout << "  " << msg[(head + i) % nmsg] << endl;
      }
    }
    cout << endl;
  }
};

}  // namespace Cengine

#endif
