



my_lib = -M lib
blas_libs = -I/opt/homebrew/opt/openblas/include -L/opt/homebrew/opt/openblas/lib -lblas
c_libs = lib/lib.c lib/lib.h
serializer_flags = --no-io-serialize-writeThis
output = --output=build
ifeq ($(fast),true)
	fast_flag = --fast
else
	fast_flag = --baseline
endif

classifier: clean
	chpl $(fast_flag) $(my_lib) $(blas_libs) -suseIOSerializers $(output)/classifier week3/classifier.chpl
	echo "Build complete."
	./build/classifier

graph: clean
	chpl $(fast_flag) $(my_lib) $(blas_libs) $(output)/graph lib/Graph.chpl
	echo "Build complete."
	./build/graph

clibs: lib/lib.c lib/lib.h
	c2chapel $(c_libs) $(output)/lib.chpl

clean: 
	rm -rf build/classifier
	rm -rf build/graph