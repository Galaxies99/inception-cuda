all:
	/usr/local/cuda-10.2/bin/nvcc -lcuda -lcublas -lineinfo conv.cu utils.cu test_conv.cu -o test_conv -Wno-deprecated-gpu-targets

clean:
	rm test_conv