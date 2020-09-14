ROOTDIR=../
include $(ROOTDIR)/Makefile.common

INCLUDE= -I$(INCLUDEDIR) -I$(SCALARDIR) -I$(TENSORDIR) -I$(ENGINEDIR) -I$(OBJECTSDIR)

DEPENDENCIES= $(INCLUDEDIR)/*.hpp $(SCALARDIR)/*.hpp $(TENSORDIR)/*.hpp $(ENGINEDIR)/*.hpp $(OBJECTSDIR)/*.hpp

TESTS= testCengine1 testCengine2 testCengine3 testCengine4 testCengine6 testCengine7 

EXECS= 

CUDA_OBJECTS=  
CUDA_EXTERNS=

ifdef WITH_CUDA
INCLUDE+=-I$(CUDA_HOME)/include
CUDA_EXTERNS+=
endif 




# -------------------------------------------------------------------------------------------------------------



testCengine1: testCengine1.cpp $(DEPENDENCIES) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testCengine1 testCengine1.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testCengine1 testCengine1.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testCengine2: testCengine2.cpp $(DEPENDENCIES) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testCengine2 testCengine2.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testCengine2 testCengine2.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testCengine3: testCengine3.cpp $(DEPENDENCIES) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testCengine3 testCengine3.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testCengine3 testCengine3.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testCengine4: testCengine4.cpp $(DEPENDENCIES) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testCengine4 testCengine4.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testCengine4 testCengine4.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testCengine5: testCengine5.cpp $(DEPENDENCIES) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testCengine5 testCengine5.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testCengine5 testCengine5.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testCengine6: testCengine6.cpp $(DEPENDENCIES) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testCengine6 testCengine6.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testCengine6 testCengine6.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testCengine7: testCengine7.cpp $(DEPENDENCIES) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testCengine7 testCengine7.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testCengine7 testCengine7.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif



objects: $(OBJECTS) 
tests: $(TESTS)
execs: $(EXECS)
derived: $(DERIVED) 

all: objects execs derived tests 

clean: 
	rm -f $(TESTS) #$(EXECS) $(OBJECTS) 

anew: clean all





