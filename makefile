


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
	chpl $(fast_flag) $(my_lib) $(blas_libs)  $(output)/classifier week5/classifier.chpl
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

speedtest:

	for n in 10 50 100 500 1000 4000 8000 12000 16000 20000 30000 40000 ; do \
		START=$$(date +%s)\
		; python3 week2/classifier.py $$n > /dev/null 2> /dev/null\
		; STOP=$$(date +%s) \
		; echo "[non-iter] Images, $$n, Execution time: $$((STOP-START))" ; \
	done

	for n in 10 50 100 500 1000 4000 8000 12000 16000 20000 30000 40000 ; do \
		START=$$(date +%s)\
		; ./build/classifier --epochs=100 --numImages=$$n --useNewIter=true > /dev/null \
		; STOP=$$(date +%s) \
		; echo "[iter] Images $$n, Execution time: $$((STOP-START))"; \
	done

	for n in 10 50 100 500 1000 4000 8000 12000 16000 20000 30000 40000 ; do \
		START=$$(date +%s)\
		; ./build/classifier --epochs=100 --numImages=$$n --useNewIter=false > /dev/null \
		; STOP=$$(date +%s) \
		; echo "[non-iter] Images, $$n, Execution time: $$((STOP-START))" ; \
	done

clean: 
	rm -rf build
	mkdir build
