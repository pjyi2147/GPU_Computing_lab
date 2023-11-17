
NVCC=nvcc
INCLUDE= -I./../libgputk
LIBgpuTK= -L./../build-dir -lgputk -lcuda
LIBS= $(LIBgpuTK)

all: template


template:
	$(NVCC) -std=c++11 template.cu $(INCLUDE) $(LIBS) -o ListScan_template


dataset_generator:
	$(NVCC) -std=c++11 dataset_generator.cpp $(INCLUDE) $(LIBS) -o dataset_generator


clean:
	rm -f ListScan_template dataset_generator
