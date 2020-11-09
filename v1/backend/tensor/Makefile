ROOTDIR=../../
include $(ROOTDIR)/Makefile.common

INCLUDE= -I$(INCLUDEDIR) -I$(SCALARDIR) -I$(TENSORDIR) -I$(ENGINEDIR) -I$(OBJECTSDIR)

DEPENDENCIES= $(INCLUDEDIR)/*.hpp $(SCALARDIR)/*.hpp $(TENSORDIR)/*.hpp $(ENGINEDIR)/*.hpp $(OBJECTSDIR)/*.hpp

TESTS=

EXECS= 

CUDA_OBJECTS=  
CUDA_EXTERNS=

ifdef WITH_CUDA
INCLUDE+=-I$(CUDA_HOME)/include
CUDA_OBJECTS+=CtensorBpack.o  
CUDA_EXTERNS+=$(INCLUDEDIR)/Cengine_base.cu 
NVCCFLAGS+= -D_WITH_CUDA -D_WITH_CUBLAS
endif 


# -------------------------------------------------------------------------------------------------------------


CtensorBpack.o: $(SO3_HEADERS) *.hpp CtensorBpack.cu Makefile  
	$(NVCC) $(NVCCFLAGS) -c CtensorBpack.cu $(CFLAGS) $(MACROS) $(INCLUDE) 

objects: $(OBJECTS) $(CUDA_OBJECTS)
tests: $(TESTS)
execs: $(EXECS)
derived: $(DERIVED) 

all: objects execs derived tests 

clean: 
	rm -f $(TESTS) $(EXECS) $(OBJECTS) 

anew: clean all


