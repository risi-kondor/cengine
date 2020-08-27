ROOTDIR=../../../
include $(ROOTDIR)/Makefile.common

BACKEND= -I$(BACKEND_SCALARDIR) -I$(BACKEND_TENSORDIR)  -I$(BACKEND_SO3DIR) 
FRONTEND= -I$(INCLUDEDIR) -I$(SCALARDIR) -I$(TENSORDIR) -I$(SO3DIR) -I$(ENGINEDIR) 

BACKEND_DEP= $(BACKEND_SCALARDIR)/*.hpp $(BACKEND_TENSORDIR)/*.hpp  $(BACKEND_SO3DIR)/*.hpp  
FRONTEND_DEP= $(INCLUDEDIR)/*.hpp $(SCALARDIR)/*.hpp $(TENSORDIR)/*.hpp $(SO3DIR)/*.hpp $(ENGINEDIR)/*.hpp  

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





