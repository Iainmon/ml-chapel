


my_lib = -M lib

UNAME := $(shell uname)
ifeq ($(UNAME), Linux)
	blas_libs = -I/chapel/home/moncrief/blas/include -L/chapel/home/moncrief/blas/lib -lopenblas
endif
ifeq ($(UNAME), Darwin)
	blas_libs = -I/opt/homebrew/opt/openblas/include -L/opt/homebrew/opt/openblas/lib -lblas
endif

c_libs = lib/lib.c lib/lib.h
serializer_flags = --no-io-serialize-writeThis
output = --output=build

ifeq ($(fast),true)
	fast_flag = --fast
else
	fast_flag = --baseline
endif

classifier: clean
	chpl $(fast_flag) $(my_lib) $(blas_libs)  $(output)/classifier week4/classifier.chpl
	echo "Build complete."
	./build/classifier

graph: clean
	chpl $(fast_flag) $(my_lib) $(blas_libs) $(output)/graph lib/Graph.chpl
	echo "Build complete."
	./build/graph

clibs: lib/lib.c lib/lib.h
	c2chapel $(c_libs) $(output)/lib.chpl

linatest: clean
	chpl $(fast_flag) $(my_lib) $(blas_libs) $(output)/LinearTest lib/LinearTest.chpl
	echo "Build complete."
	./build/LinearTest

clean: 
	rm -rf build
	mkdir build
