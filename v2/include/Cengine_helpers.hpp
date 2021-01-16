#ifndef _Cengine_helpers
#define _Cengine_helpers


namespace Cengine{

  // ---- Convenience functions ------------------------------------------------------------------------------


  template<typename TYPE>
  void stdadd(const TYPE* beg, const TYPE* end, TYPE* dest){
    const int n=end-beg; 
    for(int i=0; i<n; i++)
      dest[i]+=beg[i];
  }



}



#endif 

