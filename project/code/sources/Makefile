
NVCC=nvcc
INCLUDE= -I./../libgputk
LIBgpuTK= -L./../build -lgputk -lcuda --extended-lambda
LIBS= $(LIBgpuTK)

all: template


template:
	$(NVCC) -std=c++17 template_v6.cu $(INCLUDE) $(LIBS) -o ConvexHull_template


dataset_generator:
	$(NVCC) -std=c++11 dataset_generator.cpp $(INCLUDE) $(LIBS) -o dataset_generator


clean:
	rm -f ConvexHull_template dataset_generator
