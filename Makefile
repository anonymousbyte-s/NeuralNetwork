SRCS = main.cpp NN.cpp

main: $(SRCS)
	g++ -O3 $(SRCS) -o main