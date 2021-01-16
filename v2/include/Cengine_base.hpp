#ifndef _Cengine_base 
#define _Cengine_base 



// ---- OPTIONS ----------------------------------------------------------------------------------------------


//#define WAITING_OPT(cmd) cmd
#define WAITING_OPT(cmd)
#define WITH_FASTLIST 
#define WITH_TINYSET 


// ---- DEBUGGING -------------------------------------------------------------------------------------------


#ifdef DEBUG_ENGINE_FLAG
#define DEBUG_ENGINE(cmd) cmd;
#define DEBUG_ENGINE2(cmd) {CoutLock lk; cout<<cmd<<endl;}
#define CENGINE_ECHO_WORKERS
#define CENGINE_ECHO_QUEUE
#else
#define DEBUG_ENGINE(cmd);
#define DEBUG_ENGINE2(cmd);
#endif 

#ifdef DEBUG_FLUSH_FLAG
#define DEBUG_FLUSH(cmd) cmd;
#define DEBUG_FLUSH2(cmd) {CoutLock lk; cout<<cmd<<endl;}
#else
#define DEBUG_FLUSH(cmd);
#define DEBUG_FLUSH2(cmd);
#endif 

#ifdef CENGINE_TRACEBACK_FLAG
#define CENGINE_TRACE(msg) {traceback(msg);}
#define CENGINE_ASSERT(condition)				\
  if(!(condition)) {traceback.dump(); assert(condition);}
#define CENGINE_DUMP_TRACE() traceback.dump();
#else
#define CENGINE_ASSERT(condition);
#define CENGINE_TRACE(msg);
#define CENGINE_DUMP_TRACE(); 
#endif

#ifdef CENGINE_ECHO_WORKERS
#define CENGINE_WORKER_ECHO(cmd) {CoutLock lk; cout<<cmd<<endl;}
#else 
#define CENGINE_WORKER_ECHO(cmd);
#endif 

#ifdef CENGINE_ECHO_QUEUE
#define CENGINE_QUEUE_ECHO(cmd) {CoutLock lk; cout<<cmd<<endl;}
#else
#define CENGINE_QUEUE_ECHO(cmd);
#endif 


// ---- VERIFICATION -----------------------------------------------------------------------------------------


#ifdef CENGINE_OBJ_COUNT
#define CNODE_CREATE() ::Cengine::Cnode_count++; //cout<<::Cengine::Cnode_count<<endl;
#define CNODE_DESTROY() ::Cengine::Cnode_count--; //cout<<::Cengine::Cnode_count<<endl;
#define CHANDLE_CREATE() ::Cengine::Chandle_count++; //cout<<::Cengine::Chandle_count<<endl;
#define CHANDLE_DESTROY() ::Cengine::Chandle_count--; //cout<<::Cengine::Chandle_count<<endl;
#define COPERATOR_CREATE() ::Cengine::Coperator_count++; //cout<<::Cengine::Coperator_count<<endl;
#define COPERATOR_DESTROY() ::Cengine::Coperator_count--; //cout<<::Cengine::Coperator_count<<endl;
#define RSCALARB_CREATE() ::Cengine::RscalarB_count++; //cout<<::Cengine::RscalarB_count<<endl;
#define RSCALARB_DESTROY() ::Cengine::RscalarB_count--; //cout<<::Cengine::RscalarB_count<<endl;
#define CSCALARB_CREATE() ::Cengine::CscalarB_count++; //cout<<::Cengine::CscalarB_count<<endl;
#define CSCALARB_DESTROY() ::Cengine::CscalarB_count--; //cout<<::Cengine::CscalarB_count<<endl;
#define CTENSORB_CREATE() ::Cengine::CtensorB_count++; // {CoutLock lk; cout<<"CtensorB("<<dims<<","<<nbu<<","<<device<<")"<<endl;} 
#define CTENSORB_DESTROY() ::Cengine::CtensorB_count--; //cout<<::Cengine::CtensorB_count<<endl;
#define CMATRIXB_CREATE() ::Cengine::CmatrixB_count++; // {CoutLock lk; cout<<"CtensorB("<<dims<<","<<nbu<<","<<device<<")"<<endl;} 
#define CMATRIXB_DESTROY() ::Cengine::CmatrixB_count--; //cout<<::Cengine::CtensorB_count<<endl;
#define CTENSORARRAYB_CREATE() ::Cengine::CtensorBarray_count++; // {CoutLock lk; cout<<"CtensorB("<<dims<<","<<nbu<<","<<device<<")"<<endl;} 
#define CTENSORARRAYB_DESTROY() ::Cengine::CtensorBarray_count--; //cout<<::Cengine::CtensorB_count<<endl;
#else
#define CNODE_CREATE();
#define CNODE_DESTROY();
#define CHANDLE_CREATE();
#define CHANDLE_DESTROY();
#define COPERATOR_CREATE();
#define COPERATOR_DESTROY();
#define RSCALARB_CREATE();
#define RSCALARB_DESTROY();
#define CSCALARB_CREATE();
#define CSCALARB_DESTROY();
#define CTENSORB_CREATE();
#define CTENSORB_DESTROY();
#define CMATRIXB_CREATE();
#define CMATRIXB_DESTROY();
#define CTENSORARRAYB_CREATE();
#define CTENSORARRAYB_DESTROY();
#endif 


namespace Cengine{


  enum class engine_mode{online,biphasic};


  // ---- Multithreading ------------------------------------------------------------------------------------


  class CoutLock{
  public:
    CoutLock(): lock(mx){}
    lock_guard<mutex> lock;
    static mutex mx;
  };


  
}

#endif 
