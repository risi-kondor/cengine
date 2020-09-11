#ifndef _Cworker
#define _Cworker

#include <thread>


namespace Cengine{

  class Cengine;


  class Cworker{
  public:

    Cengine* owner;
    int id;
    bool killflag=false; 
    bool working=false; 

    thread th;

    Cworker(Cengine* _owner, const int _id): 
      owner(_owner), id(_id), 
      th([this](){this->run();}){}

    ~Cworker(){
      killflag=true; 
      th.join();
    }

	
  public:

    void run();

  };

}

#endif
