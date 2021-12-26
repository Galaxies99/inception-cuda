all:
	/usr/local/cuda-10.2/bin/nvcc -arch=sm_60 -fmad=false -lcuda -lcublas -lcudnn -lineinfo utils.cu conv.cu fc.cu pooling.cu activation.cu opers.cu layers.cu test_layer1.cu -o ../test/test_layer1 -Wno-deprecated-gpu-targets
	/usr/local/cuda-10.2/bin/nvcc -arch=sm_60 -fmad=false -lcuda -lcublas -lcudnn -lineinfo utils.cu conv.cu fc.cu pooling.cu activation.cu opers.cu layers.cu test_layer2.cu -o ../test/test_layer2 -Wno-deprecated-gpu-targets
	/usr/local/cuda-10.2/bin/nvcc -arch=sm_60 -fmad=false -lcuda -lcublas -lcudnn -lineinfo utils.cu conv.cu fc.cu pooling.cu activation.cu opers.cu layers.cu test_layer3.cu -o ../test/test_layer3 -Wno-deprecated-gpu-targets
	/usr/local/cuda-10.2/bin/nvcc -arch=sm_60 -fmad=false -lcuda -lcublas -lcudnn -lineinfo utils.cu conv.cu fc.cu pooling.cu activation.cu opers.cu layers.cu test_layer4.cu -o ../test/test_layer4 -Wno-deprecated-gpu-targets
	/usr/local/cuda-10.2/bin/nvcc -arch=sm_60 -fmad=false -lcuda -lcublas -lcudnn -lineinfo utils.cu conv.cu fc.cu pooling.cu activation.cu opers.cu layers.cu test_layer5.cu -o ../test/test_layer5 -Wno-deprecated-gpu-targets
	/usr/local/cuda-10.2/bin/nvcc -arch=sm_60 -fmad=false -lcuda -lcublas -lcudnn -lineinfo utils.cu conv.cu fc.cu pooling.cu activation.cu opers.cu layers.cu test_layer6.cu -o ../test/test_layer6 -Wno-deprecated-gpu-targets 
	/usr/local/cuda-10.2/bin/nvcc -arch=sm_60 -fmad=false -lcuda -lcublas -lcudnn -lineinfo utils.cu conv.cu fc.cu pooling.cu activation.cu opers.cu layers.cu test_outputlayer.cu -o ../test/test_outputlayer -Wno-deprecated-gpu-targets 
	/usr/local/cuda-10.2/bin/nvcc -arch=sm_60 -fmad=false -lcuda -lcublas -lcudnn -lineinfo conv.cu utils.cu test_conv.cu -o ../test/test_conv -Wno-deprecated-gpu-targets
	/usr/local/cuda-10.2/bin/nvcc -arch=sm_60 -fmad=false -lcuda -lcublas -lcudnn -lineinfo fc.cu utils.cu test_fc.cu -o ../test/test_fc -Wno-deprecated-gpu-targets 
	/usr/local/cuda-10.2/bin/nvcc -arch=sm_60 -fmad=false -lcuda -lcublas -lcudnn -lineinfo pooling.cu utils.cu test_pooling.cu -o ../test/test_pooling -Wno-deprecated-gpu-targets
	/usr/local/cuda-10.2/bin/nvcc -arch=sm_60 -fmad=false -lcuda -lcublas -lcudnn -lineinfo activation.cu test_activation.cu -o ../test/test_activation -Wno-deprecated-gpu-targets
	/usr/local/cuda-10.2/bin/nvcc -arch=sm_60 -fmad=false -lcuda -lcublas -lcudnn -lineinfo opers.cu test_opers.cu -o ../test/test_opers -Wno-deprecated-gpu-targets 
	/usr/local/cuda-10.2/bin/nvcc -arch=sm_60 -fmad=false -lcuda -lcublas -lcudnn -lineinfo utils.cu conv.cu fc.cu pooling.cu activation.cu opers.cu layers.cu inception.cu test_inception.cu -o ../test/test_inception -Wno-deprecated-gpu-targets

clean:
	rm ../test/test_layer1
	rm ../test/test_layer2
	rm ../test/test_layer3
	rm ../test/test_layer4
	rm ../test/test_layer5
	rm ../test/test_layer6
	rm ../test/test_outputlayer

	rm ../test/test_conv
	rm ../test/test_fc
	rm ../test/test_pooling
	rm ../test/test_activation
	rm ../test/test_opers
	rm ../test/test_inception
