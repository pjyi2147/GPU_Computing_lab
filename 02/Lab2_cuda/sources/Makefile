
NVCC=nvcc
INCLUDE= -I./../libgputk
LIBgpuTK= -L./../build -lgputk -lcuda
LIBS= $(LIBgpuTK)

all: template


template:
	$(NVCC) -std=c++11 template.cu $(INCLUDE) $(LIBS) -o TiledGEMM_template


dataset_generator:
	$(NVCC) -std=c++11 dataset_generator.cpp $(INCLUDE) $(LIBS) -o dataset_generator


clean:
	rm -f TiledGEMM_template dataset_generator
