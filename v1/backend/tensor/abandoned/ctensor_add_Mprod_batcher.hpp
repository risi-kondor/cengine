#ifndef _ctensor_add_Mprod_batcher
#define _ctensor_add_Mprod_batcher

// #include ""

namespace Cengine {

template <typename BATCHER>
class exec_batcher_op : public Coperator {
 public:
  Batcher* batcher;

  exec_batcher_op(Batcher* _batcher) : batcher(_batcher){};

  void exec() { batcher->exec(); }

  string str() const { return "exec_batcher_op"; }
};

class ctensor_add_Mprod_batcher : public Batcher {
 public:
  BasicCnodeEngine* engine;

  set<Cnode*> waiting;
  vector<Cnode*> ready;
  bool working = false;

  ctensor_add_Mprod_batcher(BasicCnodeEngine* _engine,
                            const ctensor_Mprod_signature& _signature)
      : engine(_engine) {}

  ~ctensor_add_Mprod_batcher() {}

 public:
  void push(Coperator* op) {
    // make sure it is not executing
    Cnode* node = op->owner;
    DEBUG_ENGINE({
      CoutLock lk;
      cout << "    Batching " << node->ident() << " [" << node->op->str()
           << "] " << endl;
    });

    if (node->nblockers == 0) {
      ready.push_back(node);
    } else {
      waiting.insert(node);
    }
    node->batcher = this;
    check_status();
  }

  void release(Cnode* node) {
    // make sure it is not executing
    DEBUG_ENGINE({
      CoutLock lk;
      cout << "    Releasing " << node->ident() << " in batcher" << endl;
    });
    ready.push_back(node);
    waiting.erase(node);
    check_status();
  }

  void kill(Cnode* node) {
    // make sure it is not executing
    DEBUG_ENGINE({
      CoutLock lk;
      cout << "    Killing " << node->ident() << " in batcher" << endl;
    });
  }

  void check_status() {
    if (ready.size() >= 3) {
      working = true;
      Cnode* node = engine->new_node(
          new exec_batcher_op<ctensor_add_Mprod_batcher>(this));
      engine->release(node);
    }
  }

  void exec() {
    DEBUG_ENGINE({
      CoutLock lk;
      cout << "    \e[1mRunning batched ctensor_add_Mprod\e[0m" << endl;
    });

    for (auto node : ready) {
      Coperator* op = node->op;
      op->exec();
      engine->done(node);

      /*
      for(int i=0; i<op->inputs.size(); i++){
        Cnode* d=op->inputs[i];
        for(int j=0; j<i; j++)
          if(op->inputs[j]==d){d=nullptr;}
        if(d!=nullptr) d->remove_dependent(node);
      }

      for(auto p: node->dependents){
        node->remove_blocker(node);
      }

      node->computed=true;
      node->working=false;
      node->batcher=nullptr;
      */
    }

    ready.clear();
    DEBUG_ENGINE({
      CoutLock lk;
      cout << "    \e[1mDone.\e[0m" << endl;
    });
  }

  int flush() {
    DEBUG_ENGINE({
      CoutLock lk;
      cout << "    \e[1mFlushing batcher.\e[0m" << endl;
    });
    working = true;
    exec();
    return waiting.size();
  }
};

}  // namespace Cengine

#endif
