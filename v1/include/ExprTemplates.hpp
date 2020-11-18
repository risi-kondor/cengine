#ifndef _ExprTemplates
#define _ExprTemplates


namespace Cengine{

  template<typename OBJ>
  class Transpose{
  public:
    const OBJ& obj;
    explicit Transpose(const OBJ& _obj):obj(_obj){}
  };

  template<typename OBJ>
  class Conjugate{
  public:
    const OBJ& obj;
    explicit Conjugate(const OBJ& _obj):obj(_obj){}
  };

  template<typename OBJ>
  class Hermitian{
  public:
    const OBJ& obj;
    explicit Hermitian(const OBJ& _obj):obj(_obj){}
  };


}

#endif
