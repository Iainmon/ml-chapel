

my_lib = -M lib
blas_libs = -I/opt/homebrew/opt/openblas/include -L/opt/homebrew/opt/openblas/lib -lblas
output = --output=build
ifeq ($(fast),true)
	fast_flag = --fast
else
	fast_flag = --baseline
endif

classifier: clean
	chpl $(fast_flag) $(my_lib) $(blas_libs) $(output)/classifier week3/classifier.chpl
	echo "Build complete."
	./build/classifier

clean: 
	rm -rf build/classifier