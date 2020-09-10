ROOTDIR=../../../
include $(ROOTDIR)/Makefile.common

BACKEND= -I$(SCALARDIR) -I$(TENSORDIR) 
FRONTEND= -I$(INCLUDEDIR) -I$(ENGINEDIR) -I$(OBJECTSDIR)  

BACKEND_DEP= $(SCALARDIR)/*.hpp $(TENSORDIR)/*.hpp  
FRONTEND_DEP= $(INCLUDEDIR)/*.hpp $(ENGINEDIR)/*.hpp  

TESTS= testCtensorBpack 

EXECS= 

CUDA_OBJECTS=  
CUDA_EXTERNS=

ifdef WITH_CUDA
INCLUDE+=-I$(CUDA_HOME)/include
CUDA_OBJECTS+=$(SO3DIR)/backendB/SO3partB.o $(SO3DIR)/backendB/SO3partBpack.o
CUDA_EXTERNS+=$(INCLUDEDIR)/GEnet_base.cu 
endif 




# -------------------------------------------------------------------------------------------------------------



testCtensorBpack: testCtensorBpack.cpp $(FRONTEND_DEP) $(BACKEND_DEP) 
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testCtensorBpack testCtensorBpack.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(FRONTEND) $(BACKEND) $(LIBS) 
else
	$(CC) -o testCtensorBpack testCtensorBpack.cpp $(CFLAGS) $(FRONTEND) $(BACKEND) $(LIBS) 
endif


objects: $(OBJECTS) 
tests: $(TESTS)
execs: $(EXECS)
derived: $(DERIVED) 

all: objects execs derived tests 

clean: 
	rm -f $(TESTS) #$(EXECS) $(OBJECTS) 

anew: clean all





