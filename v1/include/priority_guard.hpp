#ifndef _priority_guard
#define _priority_guard

//#include <deque>
//#include <chrono>



namespace Cengine{

  template<int k>
  class priority_mutex{
  public:

    atomic<bool> locked;

    vector<condition_variable> gates;
    vector<mutex> gate_mx;
    vector<int> nwaiting; 

    mutex guard_mx; 
    mutex safety_mx; 

    priority_mutex(): gates(k), gate_mx(k), nwaiting(k,0){
      locked=false; 
      safety_mx.unlock();
    }

  };


  template<int k>
  class priority_guard{
  public:

    priority_mutex<k>& mx;

    priority_guard(priority_mutex<k>& _mx, const int level=0):
      mx(_mx){

      if(true){
	lock_guard<mutex> lock(mx.guard_mx);
	if(!mx.locked){
	  mx.locked=true;
	  mx.safety_mx.lock();
	  return;
	}
	mx.nwaiting[level]++;
      }

      unique_lock<mutex> gate_lock(mx.gate_mx[0]); // only uses gate_mx[0]!
      mx.gates[level].wait_for(gate_lock,chrono::milliseconds(10),[this](){
	  //return !mx.locked;
	  lock_guard<mutex> lock(mx.guard_mx);
	  if(!mx.locked){mx.locked=true; return true;}
	  return false;
	});
      
      if(true){
	lock_guard<mutex> lock(mx.guard_mx);
	mx.nwaiting[level]--;
	//mx.locked=true;
      }
      mx.safety_mx.lock();
    }
    
    ~priority_guard(){
      mx.safety_mx.unlock();
      lock_guard<mutex> lock(mx.guard_mx);
      mx.locked=false;
      for(int i=k-1; i>=0; i--)
	if(mx.nwaiting[i]>0){
	  //unique_lock<mutex> gate_lock(mx.gate_mx[0]);
	  mx.gates[i].notify_one();
	  break;
	}
    }


  };

}

#endif 
