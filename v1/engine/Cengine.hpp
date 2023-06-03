#ifndef _Cengine
#define _Cengine

#include <chrono>
#include <deque>

#include "priority_guard.hpp"

#include "CengineHelpers.hpp"
#include "Chandle.hpp"
#include "Cnode.hpp"
#include "Cobject.hpp"
#include "Coperator.hpp"
#include "Cworker.hpp"
#include "MetaBatcher.hpp"
#include "MetaRbatcher.hpp"

namespace Cengine {

class Cengine : public BasicCnodeEngine {
 public:
  int nbatchers = 0;

  set<Cnode*> nodes;
  WAITING_OPT(set<Cnode*> waiting;);

  vector<Cworker*> workers;
  bool shutdown = false;
  bool biphasic = false;
  bool hold = false;

  int nnodes = 0;
  int nhandles = 0;

#ifdef ENGINE_PRIORITY
  priority_mutex<3> done_pmx;
#else
  mutex done_mx;
#endif

  condition_variable get_task_cv;
  mutex get_task_mx;
  atomic<int> active_workers;
  mutex active_workers_mx;
  condition_variable active_workers_cv;

  atomic<int> active_batchers;
  mutex active_batchers_mx;
  condition_variable active_batchers_cv;

  deque<Cnode*> ready;
  mutex ready_mx;
  mutex ready_list_empty_mx;
  condition_variable ready_list_empty_cv;

  vector<Batcher*> batchers;
  deque<Cnode*> ready_batchers;
  mutex ready_batchers_mx;
  mutex ready_batchers_empty_mx;
  condition_variable ready_batchers_empty_cv;
  bool batching = false;

  int nrbatchers = 0;
  vector<Rbatcher_base*> rbatchers;

  thread* sentinel;

 public:
  Cengine() : Cengine(3) {}

  Cengine(const int _nworkers) {
    active_workers = _nworkers;
    active_batchers = 0;
    for (int i = 0; i < _nworkers; i++) {
      workers.push_back(new Cworker(this, i));
    }
    if (false) {
      sentinel = new thread([this]() {
        while (true) {
          cout << ":" << active_workers << endl;
          this_thread::sleep_for(chrono::milliseconds(500));
        }
      });
    }
  }

  ~Cengine() {
    DEBUG_ENGINE({
      CoutLock lk;
      cout << "\e[1mShutting down engine.\e[0m" << endl;
    });
    CENGINE_TRACE("\e[1mShutting down engine.\e[0m");
    shutdown = true;
    get_task_cv.notify_all();
    for (auto p : workers) {
      delete p;
    }
    for (auto p : batchers) {
      delete p;
    }
    for (auto p : rbatchers) {
      delete p;
    }
    for (auto p : nodes) {
      delete p;
    }
  }

 public
     :  // ---- Nodes
        // --------------------------------------------------------------------------------------
  Cnode* new_node(Coperator* op) {
    Cnode* node = new Cnode(op);
    op->owner = node;
    node->id = nnodes++;
    return node;
  }

#include "Cengine_push_templates.hpp"

  // ---- Direct access
  // ------------------------------------------------------------------------------------

  void direct(Chandle* h, std::function<void(Cobject& obj)> f) {
    flush(h->node);
    f(*h->node->obj);
  }

  template <typename RET>
  RET direct(Chandle* h, std::function<RET(Cobject& obj)> f) {
    flush(h->node);
    return f(*h->node->obj);
  }

  // ---- Enqueue
  // ------------------------------------------------------------------------------------------

  Chandle* operator()(Coperator* op) { return enqueue_for_handle(op); }

  Chandle* enqueue_for_handle(Coperator* op) {  // Protected by done_mx
#ifdef ENGINE_PRIORITY
    priority_guard<3> lock(done_pmx, 0);
#else
    lock_guard<mutex> lock(done_mx);
#endif
    Cnode* r = enqueue_sub(op);
    Chandle* hdl = new Chandle(r);
    nhandles++;
    hdl->id = nhandles - 1;
    return hdl;
  }

  Cnode* enqueue_sub(Coperator* op) {
    Cnode* node = new_node(op);
    node->engine = this;
    Cnode* rnode = node;
    Cnode* sibling = nullptr;

    // An in-place operator is dependent on
    // all dependents of its self-argument
    if (dynamic_cast<InPlaceOperator*>(op)) {
#ifdef WITH_TINYSET
      op->inputs[0]->dependents.map([&](Cnode* p) {
        if (p->dependents.insert(node)) {
          if (!p->computed) {
            node->nblockers++;
          }
        }
      });
#else
      for (auto p : op->inputs[0]->dependents) {
        if ((p->dependents.insert(node)).second) {
          if (!p->computed) {
            node->nblockers++;
          }
        }
      }
#endif
    }

    // Delegate to batched operator
    if (dynamic_cast<BatchedOperator*>(op) && batching) {
      BatchedOperator* bop = dynamic_cast<BatchedOperator*>(op);
      if (bop->batcher_id() == 0) {
        bop->set_batcher_id(++nbatchers);
        batchers.push_back(bop->spawn_batcher());
      }
      if (dynamic_cast<InPlaceOperator*>(op)) {
        op->inputs[0]->is_view = true;
      }
#ifdef WITH_TINYSET
      for (auto p : op->inputs) {
        if (p->dependents.insert(node)) {
          if (!p->computed) {
            node->nblockers++;
          }
        }
      }
#else
      for (auto p : op->inputs) {
        if ((p->dependents.insert(node)).second) {
          if (!p->computed) {
            node->nblockers++;
          }
        }
      }
#endif
      batchers[bop->batcher_id() - 1]->push(op);
      nodes.insert(node);
      return rnode;
    }

    // Make diamond to reflect commutativity of cumulative operators
    if (dynamic_cast<CumulativeOperator*>(op)) {
      Cnode* father = op->inputs[0];
      Coperator* father_op = father->op;
      if (!father->computed && !father->working) {
        if (dynamic_cast<CumulativeOperator*>(father_op)) {
          CENGINE_QUEUE_ECHO("    Creating diamond");
          CENGINE_TRACE("Creating diamond");
          Cnode* grandfather = father->op->inputs[0];
          for (auto& p : op->inputs) {
            if (p == father) {
              p = grandfather;
            }
          }
          sibling = father;

          if (dynamic_cast<RbatchedOperator*>(op) &&
              typeid(*father_op) == typeid(*op) && !father->released) {
            rbatch_with_sibling(
                father,
                node);  // if father is released should really wait on it
          }
        }

        if (dynamic_cast<diamond_op*>(father_op) && !father->released) {
          CENGINE_QUEUE_ECHO("    (Extending diamond)");
          CENGINE_TRACE("(Extending diamond)");
          Cnode* greatgrandfather =
              father->op->inputs[0]->op->inputs[0];  // father()->father();
          for (auto& p : op->inputs) {
            if (p == father) {
              p = greatgrandfather;
            }
          }
          node->dependents.insert(father);
          node->is_view = true;
          father_op->inputs.push_back(node);
          father->nblockers++;
          rnode = father;

          if (dynamic_cast<RbatchedOperator*>(op)) {
            for (int i = 0; i < father_op->inputs.size() - 1; i++) {
              Cnode* grandfather = father_op->inputs[i];
              if (typeid(*grandfather->op) == typeid(*op) &&
                  !grandfather->released) {
                rbatch_with_sibling(grandfather, node);
                break;
              }
            }
          }
        }
      }
    }

    if (dynamic_cast<InPlaceOperator*>(op)) {
      op->inputs[0]->is_view = true;
    }

    for (auto p : op->inputs) {
#ifdef WITH_TINYSET
      if (p->dependents.insert(node)) {  // changed
        if (!p->computed) {
          node->nblockers++;
        }  // COUT(">>"<<p->ident());}
      }
#else
      if ((p->dependents.insert(node)).second) {  // changed
        if (!p->computed) {
          node->nblockers++;
        }  // COUT(">>"<<p->ident());}
      }
#endif
    }

    // Complete diamond
    if (sibling) {
      Cnode* nnode = enqueue_sub(new diamond_op(sibling, node));
      // nnode->obj=node->obj;
      node->is_view = true;
      sibling->is_view = true;
      rnode = nnode;
    }

    if (!node->rbatcher) {
      if (node->nblockers == 0) {
        release(node);
        CENGINE_QUEUE_ECHO("    Early   " << node->ident() << " ["
                                          << node->op->str() << "]");
        CENGINE_TRACE("Early      " + node->ident() + " [" + node->op->str() +
                      "] ");
      } else {
        WAITING_OPT(waiting.insert(node););
        CENGINE_QUEUE_ECHO("    Queuing " << node->ident() << " ["
                                          << node->op->str() << "]");
        CENGINE_TRACE("Queuing    " + node->ident() + " [" + node->op->str() +
                      "] ");
      }
    } else {
      if (node->nblockers == 0) {
        node->rbatcher->release(node);
      }
    }

    nodes.insert(node);
    return rnode;
  }

  // ---- Releasing nodes
  // ----------------------------------------------------------------------------------

  void release(Cnode* node) {  // visited by workers but protected by done_mx
    // DEBUG_ENGINE2("    Releasing "<<node->ident());

    if (node->dependents.size() >
        0) {  // && dynamic_cast<diamond_op>(node->dependents[0]->op)){
      diamond_op* diamond = nullptr;
      node->dependents.map([&diamond](Cnode* n) {
        // COUT(n->op->str())
        if (dynamic_cast<diamond_op*>(n->op)) {
          diamond = dynamic_cast<diamond_op*>(n->op);
        }
      });
      if (diamond && dynamic_cast<RbatchedOperator*>(node->op) == nullptr) {
        // COUT("Hold siblings!");
        for (auto p : diamond->inputs) {
          if (p != node && !p->released && !p->computed) {
            if (p->working) {
              // if(p->dependents.insert(node)) node->nblockers++; // TODO
            } else {
              if (node->dependents.insert(p)) {
                p->nblockers++;
              }
            }
          }
        }
      }
    }

    WAITING_OPT(if (waiting.find(node) != waiting.end()) waiting.erase(node););
    node->released = true;

    {
      lock_guard<mutex> lock(ready_mx);
      auto it = find(ready.begin(), ready.end(), node);
      if (it != ready.end()) {
        ready.erase(it);
      }
      ready.push_back(node);
    }
    if (!hold) {
      get_task_cv.notify_one();
    }
  }

  void release_batcher(Cnode* node) {  // protected by done_mx
    // DEBUG_ENGINE2("    Releasing "<<node->ident());
    node->released = true;
    {
      lock_guard<mutex> lock(ready_batchers_mx);
      auto it = find(ready_batchers.begin(), ready_batchers.end(), node);
      if (it != ready_batchers.end()) {
        CoutLock lk;
        cout << "Batcher already released." << endl;
        ready_batchers.erase(it);
      }
      ready_batchers.push_back(node);
    }
    if (!hold) {
      get_task_cv.notify_one();
    }
  }

  // ---- Finishing operators
  // ------------------------------------------------------------------------------

  void done(Cnode* node) {  // visited by workers
#ifdef ENGINE_PRIORITY
    priority_guard<3> lock(done_pmx, 1);
#else
    lock_guard<mutex> lock(done_mx);
#endif

    // DEBUG_ENGINE2("    Done "<<node->ident());
    // waiting.erase(node);

    Coperator* op = node->op;
    if (op) {
      for (int i = 0; i < op->inputs.size(); i++) {
        Cnode* p = op->inputs[i];
        for (int j = 0; j < i; j++) {
          if (op->inputs[j] == p) {
            p = nullptr;
          }
        }
        if (p != nullptr) {
          // p->remove_dependent(node);
#ifdef WITH_TINYSET
          if (!p->dependents.find(node)) {
            COUT("\e[1mDependent not found \e[0m" << p->ident() << " "
                                                  << node->op->str())
          }
#else
          if (p->dependents.find(node) == p->dependents.end()) {
            COUT("\e[1mDependent not found \e[0m" << p->ident() << " "
                                                  << node->op->str())
          }
#endif
          p->dependents.erase(node);
          if (p->dependents.size() == 0 &&
              p->nhandles == 0) {  // cout<<"ss"<<endl;
            if (p->batcher) {
              p->batcher->kill(p);
            } else {
              kill(p);
            }
          }
        }
      }
    }

#ifdef WITH_TINYSET
    node->dependents.map([&](Cnode* p) {
      // p->remove_blocker(node);
      p->nblockers--;
      if (p->nblockers == 0) {
        if (p->batcher) {
          p->batcher->release(p);
        } else {
          if (p->rbatcher) {
            p->rbatcher->release(p);
          } else {
            release(p);
          }
        }
      }
    });
#else
    for (auto p : node->dependents) {
      // p->remove_blocker(node);
      p->nblockers--;
      if (p->nblockers == 0) {
        if (p->batcher) {
          p->batcher->release(p);
        } else {
          if (p->rbatcher) {
            p->rbatcher->release(p);
          } else {
            release(p);
          }
        }
      }
    }
#endif

    node->computed = true;
    node->working = false;

    if (dynamic_cast<BatcherExecutor*>(op)) {
      delete node;  // changed!
      node = nullptr;
      {
        lock_guard<mutex> lock(active_batchers_mx);
        active_batchers--;
      }  // why was this wrong?
      if (active_batchers == 0) {
        active_batchers_cv.notify_one();
      }
    }

    if (node && node->dependents.size() == 0 &&
        node->nhandles == 0) {  // may lead to orphan nodes
      kill(node);
    }

    if (ready.size() == 0) {
      ready_list_empty_cv.notify_one();
    }

    if (ready_batchers.size() == 0) {
      ready_batchers_empty_cv.notify_one();
    }
  }

  void kill(Cnode* node) {
    CENGINE_QUEUE_ECHO("    Killing " << node->ident());
    CENGINE_TRACE("Killing " + node->ident());
    return;

    if (node->dependents.size() > 0) {
      CENGINE_DUMP_TRACE();
      CoutLock lk;
      cout << "Caught dependent" << endl;
      exit(-1);
    }

    if (node->nhandles > 0) {
      CENGINE_DUMP_TRACE();
      CoutLock lk;
      cout << "Caught handle" << endl;
      exit(-1);
      return;
    }

    {
      lock_guard<mutex> lock(ready_mx);
      auto it = find(ready.begin(), ready.end(), node);
      if (it != ready.end()) {
        ready.erase(it);
      }
    }

    if (node->working || !node->computed) {
      //{CoutLock lk; cout<<"Caught working N"<<node->id<<endl;}
      // exit(-1);
      // tokill.insert(node);
      return;
    }

    if (nodes.find(node) == nodes.end()) {
      CENGINE_DUMP_TRACE();
      CoutLock lk;
      cout << "Cannot find node " << endl;
      cout << "N" << node->id << endl;
      exit(-1);
    }

    nodes.erase(node);
    // COUT("deleting"<<node->ident());
    delete node;
  }

  // ---- Flushing
  // -----------------------------------------------------------------------------------------

  void dump_batchers() {
    DEBUG_FLUSH({
      CoutLock lk;
      cout << "Dumping batchers..." << endl;
    });
#ifdef ENGINE_PRIORITY
    priority_guard<3> lock(done_pmx, 0);
#else
    lock_guard<mutex> lock(done_mx);
#endif
    for (auto p : batchers) {
      p->flush();
    }
    for (auto p : rbatchers) {
      p->flush();
    }
  }

  void flush(Cnode* node) { flush(); }

  void flush() {  // not protected by done_mx
    CENGINE_QUEUE_ECHO(endl << "    \e[1mFlushing engine...\e[0m");
    CENGINE_TRACE("\e[1mFlushing engine...\e[0m");
    int h = 0;

    if (hold) {
      hold = false;
      get_task_cv.notify_all();
    }

    bool all_done = false;
    while (true) {
      all_done = true;

      dump_batchers();

      if (ready_batchers.size() > 0) {
        DEBUG_FLUSH2("Flushing " << ready_batchers.size()
                                 << " batchers on ready list");
        unique_lock<mutex> lock(ready_batchers_empty_mx);
        ready_batchers_empty_cv.wait(
            lock, [this]() { return ready_batchers.size() == 0; });
      }

      while (ready.size() > 0) {
        DEBUG_FLUSH2("Flushing " << ready.size()
                                 << " operations on ready list");
        unique_lock<mutex> lock(ready_list_empty_mx);
        ready_list_empty_cv.wait(lock, [this]() { return ready.size() == 0; });
      }

      if (true) {
        unique_lock<mutex> block(active_batchers_mx);
        active_batchers_cv.wait(block, [this]() {
          if (active_batchers > 0) {
            DEBUG_FLUSH2("Waiting for " << active_batchers
                                        << " active batchers");
          }
          return active_batchers == 0;
        });
      }

      if (true) {
        unique_lock<mutex> wlock(active_workers_mx);
        active_workers_cv.wait(wlock, [this]() {
          if (active_workers > 0) {
            DEBUG_FLUSH2("Waiting for " << active_workers << " workers");
          }
          return active_workers == 0;
        });
      }

      // DEBUG_FLUSH2(".");

      for (auto p : batchers) {
        if (p->npending() > 0) {
          // cout<<"a:"<<p->npending()<<endl;
          all_done = false;
          break;
        }
      }
      for (auto p : rbatchers) {
        if (p->npending() > 0) {
          // cout<<"a:"<<p->npending()<<endl;
          all_done = false;
          break;
        }
      }
      if (ready.size() > 0) {
        all_done = false;
      }
      if (ready_batchers.size() > 0) {
        all_done = false;
      }
      WAITING_OPT(if (waiting.size() > 0) all_done = false;)
      if (all_done) {
        break;
      }
      // COUT(ready.size()<<" "<<ready_batchers.size()<<" "<<waiting.size());

      if (h++ > 100) {
        CoutLock lk;
        cout << "Timeout. " << endl;
        WAITING_OPT(for (auto p : waiting) cout << p->str() << endl;)
        cout << "---" << endl;
        for (auto p : ready) {
          cout << p->str() << endl;
        }
        exit(0);
      }

      // this_thread::sleep_for(chrono::milliseconds(13));
    }

    while (active_workers > 0) {
      //{CoutLock lk; cout<<"active workers"<<endl;}
      /*
      bool all_done=true;
      for(auto p:workers){
        if(p->working) all_done=false;
      }
      if(all_done) break;
      */
      this_thread::sleep_for(chrono::milliseconds(2));
    }

    if (biphasic) {
      hold = true;
    }
    DEBUG_FLUSH2("done.");
    CENGINE_QUEUE_ECHO("    \e[1mFlushed.\e[0m")
    CENGINE_TRACE("\e[1mFlushed.\e[0m")
    return;
  }

 public
     :  // ---- Handles
        // ------------------------------------------------------------------------------------
  Chandle* new_handle(Cnode* node) {
    Chandle* hdl = new Chandle(node);
    nhandles++;
    hdl->id = nhandles - 1;
    return hdl;
  }

  Chandle* new_handle(Chandle* h) { return h; }

  void dec_handle(Cnode* node) {
#ifdef ENGINE_PRIORITY
    priority_guard<3> lock(done_pmx, 1);
#else
    lock_guard<mutex> lock(done_mx);
#endif
    node->nhandles--;
    // DEBUG_ENGINE2(node->ident()<<" nh="<<node->nhandles);
    if (node->dependents.size() == 0 && node->nhandles == 0) {
      if (node->batcher) {
        node->batcher->kill(node);
      } else {
        kill(node);
      }
    }
  }

  // void kill(Chandle* hdl){
  //  what about checking node??
  // handles.erase(hdl);
  // delete hdl;
  //}

 public
     :  // ---- Rbatching
        // ----------------------------------------------------------------------------------
  void rbatch_with_sibling(Cnode* sibling, Cnode* node) {
    if (sibling->rbatcher) {
      sibling->rbatcher->push(node);
      return;
    }

    RbatchedOperator* bop = dynamic_cast<RbatchedOperator*>(node->op);
    Rbatcher_base* meta;
    if (bop->rbatcher_id() == 0) {
      bop->set_rbatcher_id(++nrbatchers);
      meta = bop->spawn_rbatcher(this);
      rbatchers.push_back(meta);
    }

    meta = rbatchers[bop->rbatcher_id() - 1];
    meta->push(sibling);
    WAITING_OPT(waiting.erase(sibling);)
    sibling->rbatcher->push(node);
  }

 public
     :  // ---- Backend
        // ------------------------------------------------------------------------------------
  Coperator* get_task(Cworker* worker) {  // visited by workers
    Coperator* op;

    worker->working = false;
    {
      lock_guard<mutex> lock(active_workers_mx);
      active_workers--;
    }  // probably don't need lock
    if (active_workers == 0) {
      active_workers_cv.notify_one();
    }

    unique_lock<mutex> lock(get_task_mx);
    get_task_cv.wait(lock, [this]() {
      return (!hold && (ready.size() > 0 || ready_batchers.size() > 0)) ||
             shutdown;
    });

    {
#ifdef ENGINE_PRIORITY
      priority_guard<3> lock(done_pmx, 2);
#else
      lock_guard<mutex> lock(done_mx);
#endif
      lock_guard<mutex> lock2(ready_mx);

      {
        lock_guard<mutex> lock(active_workers_mx);
        active_workers++;
      }

      // this_thread::sleep_for(chrono::milliseconds(0));

      if (ready_batchers.size() > 0) {
        worker->working = true;
        op = ready_batchers.front()->op;
        ready_batchers.pop_front();
        op->owner->working = true;
        if (!hold) {
          get_task_cv.notify_one();
        }
        active_batchers++;
        return op;
      }

      if (ready.size() > 0) {
        // CoutLock lk;
        // for(auto p:ready) cout<<p->str()<<" ";
        // cout<<endl;
        worker->working = true;
        op = ready.front()->op;
        ready.pop_front();
        op->owner->working = true;
        if (!hold) {
          get_task_cv.notify_one();
        }
        return op;
      }

      return nullptr;
    }
  }
};

// ---- Functions
// -----------------------------------------------------------------------------------------

inline void Cworker::run() {
  while (!owner->shutdown) {
    Coperator* op = owner->get_task(this);
    if (op) {
      CENGINE_WORKER_ECHO("    \e[1mWorker " << id << ":\e[0m  "
                                             << op->owner->ident() << " <- "
                                             << op->str());
      CENGINE_TRACE("\e[1mWorker " + to_string(id) + ":\e[0m  " +
                    op->owner->ident() + " <- " + op->str());
#ifndef CENGINE_DRY_RUN
      op->exec();
#else
      // this_thread::sleep_for(chrono::milliseconds(10));
      if (dynamic_cast<BatcherExecutor*>(op)) {
        op->exec();
      }
#endif
      CENGINE_WORKER_ECHO("    \e[1mWorker " << id << ":\e[0m  "
                                             << op->owner->ident() << " done.");
      owner->done(op->owner);
    }
  }
}

}  // namespace Cengine

#endif
