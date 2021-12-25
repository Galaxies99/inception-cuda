all:
	/usr/local/cuda-10.2/bin/nvcc -arch=sm_60 -fmad=false -lcuda -lcublas -lineinfo utils.cu conv.cu fc.cu pooling.cu activation.cu opers.cu layers.cu test_layer2.cu -o ./test_layer2 -Wno-deprecated-gpu-targets
	/usr/local/cuda-10.2/bin/nvcc -arch=sm_60 -fmad=false -lcuda -lcublas -lineinfo utils.cu conv.cu fc.cu pooling.cu activation.cu opers.cu layers.cu test_layer5.cu -o ./test_layer5 -Wno-deprecated-gpu-targets
	/usr/local/cuda-10.2/bin/nvcc -arch=sm_60 -fmad=false -lcuda -lcublas -lineinfo utils.cu conv.cu fc.cu pooling.cu activation.cu opers.cu layers.cu test_layer6.cu -o ./test_layer6 -Wno-deprecated-gpu-targets 
	/usr/local/cuda-10.2/bin/nvcc -arch=sm_60 -fmad=false -lcuda -lcublas -lineinfo conv.cu utils.cu test_conv.cu -o test_conv -Wno-deprecated-gpu-targets
	/usr/local/cuda-10.2/bin/nvcc -arch=sm_60 -fmad=false -lcuda -lcublas -lineinfo fc.cu utils.cu test_fc.cu -o test_fc -Wno-deprecated-gpu-targets 
	/usr/local/cuda-10.2/bin/nvcc -arch=sm_60 -fmad=false -lcuda -lcublas -lineinfo pooling.cu utils.cu test_pooling.cu -o test_pooling -Wno-deprecated-gpu-targets
	/usr/local/cuda-10.2/bin/nvcc -arch=sm_60 -fmad=false -lcuda -lcublas -lineinfo activation.cu test_activation.cu -o test_activation -Wno-deprecated-gpu-targets
	/usr/local/cuda-10.2/bin/nvcc -arch=sm_60 -fmad=false -lcuda -lcublas -lineinfo opers.cu test_opers.cu -o test_opers -Wno-deprecated-gpu-targets 

clean:
	rm test_layer2
	rm test_layer5
	rm test_layer6
	rm test_conv
	rm test_fc
	rm test_pooling
	rm test_activation
	rm test_opers
