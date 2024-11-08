GCC=g++
#GCC=/cvmfs/sft.cern.ch/lcg/releases/gcc/9.2.0/x86_64-centos7/bin/g++
CXXFLAGS=`root-config --libs --cflags` -lRooFit -lRooFitCore -lMinuit2 -O3 -I/usr/include/boost/ -L/usr/lib64/ -lboost_program_options -I/usr/include/eigen3

SRCDIR=.
BINDIR=.

.PHONY: all
all:
	$(info, "--- Full compilation --- ")	
	$(info, "-> if you want just to recompile something use 'make fast' ")	
	$(info, "------------------------ ")	
	$(MAKE) massfit resolfit massscales_data 

massfit: massfit.cpp
	$(GCC) $(CXXFLAGS) -o $(BINDIR)/massfit massfit.cpp  

resolfit: resolfit.cpp
	$(GCC) $(CXXFLAGS) -o $(BINDIR)/resolfit resolfit.cpp  

massscales_data: massscales_data.cpp
	$(GCC) $(CXXFLAGS) -o $(BINDIR)/massscales_data massscales_data.cpp  
