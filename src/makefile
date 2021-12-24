all:
	/usr/local/cuda-10.2/bin/nvcc -lcuda -lcublas -lineinfo conv.cu utils.cu test_conv.cu -o test_conv -Wno-deprecated-gpu-targets
	/usr/local/cuda-10.2/bin/nvcc -lcuda -lcublas -lineinfo fc.cu utils.cu test_fc.cu -o test_fc -Wno-deprecated-gpu-targets
	/usr/local/cuda-10.2/bin/nvcc -lcuda -lcublas -lineinfo activation.cu test_activation.cu -o test_activation -Wno-deprecated-gpu-targets

clean:
	rm test_conv
	rm test_fc
	rm test_activation
