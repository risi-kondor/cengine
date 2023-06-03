#ifndef _BatcherG
#define _BatcherG

#include "BatcherA.hpp"
#include "Cnode.hpp"

namespace Cengine {

// class BatcherG;
class GatherGroup;

class GatheringBatchedOperator : public BatchedOperator {
 public:
  // virtual void batched_exec(BasicCnodeEngine* engine, const
  // vector<GatherGroup*>& ggroup, const vector<Cnode*>& nodes)=0;
  virtual void batched_exec(const vector<GatherGroup*>& ggroup,
                            const vector<Cnode*>& nodes) = 0;
};

class GatherGroup {
 public:
  // BatcherG* owner;
  set<Cnode*> waiting;
  vector<Cnode*> ready;
  Cnode* target;

  GatherGroup() {}

  // GatherGroup(BatcherG* _owner): owner(_owner){}

  void push(Cnode* node) {
    node->ggroup = this;
    if (node->nblockers == 0) {
      ready.push_back(node);
    } else {
      waiting.insert(node);
    }
  }

  bool isready() const { return (waiting.size() == 0); }

  int nready() const { return ready.size(); }
};

class exec_gbatcher_op : public Coperator, public BatcherExecutor {
 public:
  BasicCnodeEngine* engine;
  vector<GatherGroup*> ggroups;
  vector<Cnode*> nodes;

  exec_gbatcher_op(BasicCnodeEngine* _engine,
                   const vector<GatherGroup*>& _ggroups,
                   const vector<Cnode*>& _nodes)
      : engine(_engine), ggroups(_ggroups), nodes(_nodes){};

  void exec() {
    if (nodes.size() == 0 && ggroups.size() == 0) {
      CoutLock lk;
      cout << "\e[1mEmpty batcherg\e[0m" << endl;
      return;
    }
    if (nodes.size() > 0) {
      dynamic_cast<GatheringBatchedOperator*>(nodes[0]->op)
          ->batched_exec(ggroups, nodes);
    } else {
      if (ggroups.size() > 0 && ggroups[0]->ready.size() > 0) {
        dynamic_cast<GatheringBatchedOperator*>(ggroups[0]->ready[0]->op)
            ->batched_exec(ggroups, nodes);
      }
    }
  }

  string str() const {
    if (nodes.size() > 0) {
      return "exec_gbatcher_op<" +
             dynamic_cast<BatchedOperator*>(nodes[0]->op)->batcher_name() + ">";
    }
    if (ggroups.size() > 0 && ggroups[0]->ready.size() > 0) {
      return "exec_gbatcher_op<" +
             dynamic_cast<BatchedOperator*>(ggroups[0]->ready[0]->op)
                 ->batcher_name() +
             ">";
    }
    return "exec_gbatcher_op<>";
  }
};

class BatcherG : public Batcher {
 public:
  BasicCnodeEngine* engine;
  string name;

  set<Cnode*> waiting;
  set<Cnode*> ready;
  // bool working=false;
  // mutex mx;

  set<GatherGroup*> ggroups;

  BatcherG(BasicCnodeEngine* _engine) : engine(_engine) {}

  BatcherG(BasicCnodeEngine* _engine, const string _name)
      : engine(_engine), name(_name) {}

  ~BatcherG() {}

 public:
  void push(Coperator* op) {  // protected by done_mx
    Cnode* node = op->owner;
    DEBUG_ENGINE({
      CoutLock lk;
      cout << "    Batching " << node->ident() << " [" << node->op->str()
           << "] " << endl;
    });
    node->batcher = this;

    if (op->inputs.size() > 0 && op->inputs[0]->batcher == this) {
      GatherGroup* gp;
      Cnode* father = op->inputs[0];

      if (father->ggroup) {
        {
          CoutLock lk;
          cout << "Extending gather." << endl;
        }
        gp = father->ggroup;
      } else {
        {
          CoutLock lk;
          cout << "Creating new gather." << endl;
        }
        gp = new GatherGroup();
        gp->target = father->op->inputs[0];
        gp->push(father);
        waiting.erase(father);
        ready.erase(father);
        ggroups.insert(gp);
      }

      Cnode* target = gp->target;
      for (auto& p : op->inputs) {  // lots of assumptions
        if (p == father) {
          p = gp->target;
        }
      }
      if (target->computed) {
        node->nblockers--;
      }
      target->dependents.insert(node);
      father->dependents.erase(node);
      gp->push(node);

    } else {
      // add solo node
      {
        CoutLock lk;
        cout << "Adding solo node." << endl;
      }
      if (node->nblockers == 0) {
        ready.insert(node);
      } else {
        waiting.insert(node);
      }
    }

    check_status();
  }

  void release(Cnode* node) {  // protected by done_mx
    DEBUG_ENGINE({
      CoutLock lk;
      cout << "    Releasing " << node->ident() << " in batcher" << endl;
    });

    GatherGroup* gp = node->ggroup;
    if (gp) {
      gp->ready.push_back(node);
      gp->waiting.erase(node);
    } else {
      ready.insert(node);
      waiting.erase(node);
    }
    check_status();
  }

  void kill(Cnode* node) {
    DEBUG_ENGINE({
      CoutLock lk;
      cout << "    Killing " << node->ident() << " in batcher" << endl;
    });
    // CoutLock lk;
    // cout<<"\e[1mKill "<<node->ident()<<" \e[0m"<<endl;
  }

  void check_status() {  // protected by done_mx
    int t = ready.size();
    for (auto p : ggroups) {
      if (p->isready()) {
        t += p->nready();
      }
    }
    if (t >= 64) {
      release();
    }
  }

  void release() {  // protected by done_mx
    DEBUG_ENGINE({
      CoutLock lk;
      cout << "  Releasing gbatcher " << name << endl;
    });
    // working=true;

    vector<GatherGroup*> groups;
    for (auto p : ggroups) {
      if (p->isready()) {
        groups.push_back(p);
      }
    }

    vector<Cnode*> readylist;
    for (auto p : ready) {
      readylist.push_back(p);
    }

    {
      CoutLock lk;
      cout << "[";
      for (auto p : groups) {
        cout << p->ready.size() << ",";
      }
      cout << readylist.size() << "]" << endl;
    }

    Cnode* node =
        engine->new_node(new exec_gbatcher_op(engine, groups, readylist));
    engine->release_batcher(node);

    for (auto p : groups) {
      for (auto q : p->ready) {
        q->batcher = nullptr;
        q->ggroup = nullptr;
      }
      ggroups.erase(p);
    }

    for (auto q : readylist) {
      q->batcher = nullptr;
    }
    ready.clear();
  }

  int flush() {  // protected_by done_mx
    // DEBUG_ENGINE({CoutLock lk; cout<<"    Flushing batcher "<<name<<".
    // "<<waiting.size()<<" "<<ready.size()<<endl;});
    int t = ready.size();
    for (auto p : ggroups) {
      if (p->isready()) {
        t += p->nready();
      }
    }
    if (t > 0) {
      release();
    }
    return waiting.size();
  }

  int npending() const { return waiting.size() + ready.size(); }
};

}  // namespace Cengine

#endif
