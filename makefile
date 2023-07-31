


my_lib = -M lib

UNAME := $(shell uname)
ifeq ($(UNAME), Linux)
	python_bin = /usr/bin/python3
	blas_libs = -I/chapel/home/moncrief/blas/include -L/chapel/home/moncrief/blas/lib -lopenblas
endif
ifeq ($(UNAME), Darwin)
	python_bin = /opt/homebrew/bin/python3
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

torch: clean
	chpl $(fast_flag) $(my_lib) $(blas_libs)  $(output)/Torch week7/Torch.chpl
	echo "Build complete."
	./build/Torch

torchClassifier: clean
	chpl $(fast_flag) $(my_lib) $(blas_libs)  $(output)/torchClassifier week6/torchClassifier.chpl
	echo "Build complete."
	./build/torchClassifier

cnn: clean
	chpl $(fast_flag) $(my_lib) $(blas_libs)  $(output)/cnn week7/cnn.chpl
	echo "Build complete."
	./build/cnn


tensor: clean
	chpl $(fast_flag) $(my_lib) $(blas_libs)  $(output)/Tensor week7/Tensor.chpl
	echo "Build complete."
	./build/Tensor



classifier: clean
	chpl $(fast_flag) $(my_lib) $(blas_libs)  $(output)/classifier week5/classifier.chpl
	echo "Build complete."
	./build/classifier

runClassifier: clean
	chpl $(fast_flag) $(my_lib) $(blas_libs)  $(output)/runClassifier week5/runClassifier.chpl
	echo "Build complete."
	./build/runClassifier

enhancedMap: clean
	chpl $(fast_flag) $(my_lib) $(blas_libs)  $(output)/enhancedMap week5/enhancedMap.chpl
	echo "Build complete."
	./build/enhancedMap

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

speedtest:

	for n in 10 50 100 500 1000 4000 8000 12000 16000 20000 30000 40000 ; do \
		START=$$(date +%s)\
		; $(python_bin) week2/classifier.py $$n > /dev/null 2> /dev/null\
		; STOP=$$(date +%s) \
		; echo "[python] Images: $$n, Execution time: $$((STOP-START))" ; \
	done

	for n in 10 50 100 500 1000 4000 8000 12000 16000 20000 30000 40000 ; do \
		START=$$(date +%s)\
		; ./build/classifier --epochs=100 --numImages=$$n --useNewIter=true > /dev/null \
		; STOP=$$(date +%s) \
		; echo "[iter] Images: $$n, Execution time: $$((STOP-START))"; \
	done

	for n in 10 50 100 500 1000 4000 8000 12000 16000 20000 30000 40000 ; do \
		START=$$(date +%s)\
		; ./build/classifier --epochs=100 --numImages=$$n --useNewIter=false > /dev/null \
		; STOP=$$(date +%s) \
		; echo "[non-iter] Images: $$n, Execution time: $$((STOP-START))" ; \
	done

tests: clean
	chpl $(fast_flag) $(my_lib) -M week5 $(blas_libs) $(output)/week5_classifier tests/week5_classifier.chpl
	chpl $(fast_flag) $(my_lib) -M week6 $(blas_libs) $(output)/week6_classifier tests/week6_classifier.chpl
	echo "Build complete."


clean: 
	rm -rf build
	mkdir build

monster:
	./build/classifier --numImages=59000  --epochs=12000 --testSize=100 | tee bigtrain.txt