// CS124 Programming Assignment 1 - Elizabeth Zhong, Helen Xiao

#include <iostream>
using namespace std;
#include <stdlib.h>

int main(int argc, char* argv[])
{
    // Check the number of parameters
    if (argc != 5) {
        // Tell the user how to run the program
        fprintf(stderr, "Usage: ./randmst 0 Numpoints Numtrials Dimension");
        return 1;
    }
    
    int n = stoi(argv[2]);
    int trials = stoi(argv[3]);
    int dim = stoi(argv[4]);
}