/*******************************************************************************
                CS124 Programming Set 1 - Elizabeth Zhong, Helen Xiao 
*******************************************************************************/
#include <iostream>
using namespace std;
#include <stdlib.h>
#include <set>
#include<vector>

// define new type, "edge" (edge weight, fst endpoint, snd endpoint)
typedef std::tuple<double, int, int> edge;

/*------------------------------------------------------------------------------
                              Merge Sort with Vectors
------------------------------------------------------------------------------*/

// Modified from Source: https://slaystudy.com/c-merge-sort-vector/ 

void MergeSortedIntervals(vector<edge>& v, int s, int m, int e) {
    // temp is used to temporary store the vector obtained by merging
    // elements from [s to m] and [m+1 to e] in v
	vector<edge> temp;

	int i, j;
	i = s;
	j = m + 1;

	while (i <= m && j <= e) {
		if (get<0>(v[i]) <= get<0>(v[j])) {
			temp.push_back(v[i]);
			++i;
		}
		else {
			temp.push_back(v[j]);
			++j;
		}
	}
	while (i <= m) {
		temp.push_back(v[i]);
		++i;
	}
	while (j <= e) {
		temp.push_back(v[j]);
		++j;
	}

	for (int i = s; i <= e; ++i)
		v[i] = temp[i - s];
}

// the MergeSort function
// Sorts the array in the range [s to e] in v using
// merge sort algorithm
void MergeSort(vector<edge>& v, int s, int e) {
	if (s < e) {
		int m = (s + e) / 2;
		MergeSort(v, s, m);
		MergeSort(v, m + 1, e);
		MergeSortedIntervals(v, s, m, e);
	}
}

/*------------------------------------------------------------------------------
                                  Merge Sort 
------------------------------------------------------------------------------*/

// Source: https://www.geeksforgeeks.org/cpp-return-2d-array-from-function/

void merge(edge array[], int const left, int const mid,
           int const right)
{
    auto const subArrayOne = mid - left + 1;
    auto const subArrayTwo = right - mid;
  
    // Create temp arrays
    auto *leftArray = new edge[subArrayOne],
         *rightArray = new edge[subArrayTwo];
  
    // Copy data to temp arrays leftArray[] and rightArray[]
    for (auto i = 0; i < subArrayOne; i++)
        leftArray[i] = array[left + i];
    for (auto j = 0; j < subArrayTwo; j++)
        rightArray[j] = array[mid + 1 + j];
  
    auto indexOfSubArrayOne
        = 0, // Initial index of first sub-array
        indexOfSubArrayTwo
        = 0; // Initial index of second sub-array
    int indexOfMergedArray
        = left; // Initial index of merged array
  
    // Merge the temp arrays back into array[left..right]
    while (indexOfSubArrayOne < subArrayOne
           && indexOfSubArrayTwo < subArrayTwo) {
        if (get<0>(leftArray[indexOfSubArrayOne])
            <= get<0>(rightArray[indexOfSubArrayTwo])) {
            array[indexOfMergedArray]
                = leftArray[indexOfSubArrayOne];
            indexOfSubArrayOne++;
        }
        else {
            array[indexOfMergedArray]
                = rightArray[indexOfSubArrayTwo];
            indexOfSubArrayTwo++;
        }
        indexOfMergedArray++;
    }
    // Copy the remaining elements of
    // left[], if there are any
    while (indexOfSubArrayOne < subArrayOne) {
        array[indexOfMergedArray]
            = leftArray[indexOfSubArrayOne];
        indexOfSubArrayOne++;
        indexOfMergedArray++;
    }
    // Copy the remaining elements of
    // right[], if there are any
    while (indexOfSubArrayTwo < subArrayTwo) {
        array[indexOfMergedArray]
            = rightArray[indexOfSubArrayTwo];
        indexOfSubArrayTwo++;
        indexOfMergedArray++;
    }
    delete[] leftArray;
    delete[] rightArray;
}
  
// begin is for left index and end is
// right index of the sub-array
// of arr to be sorted */
void mergeSort(edge array[], int const begin, int const end)
{
    if (begin >= end)
        return; // Returns recursively
  
    auto mid = begin + (end - begin) / 2;
    mergeSort(array, begin, mid);
    mergeSort(array, mid + 1, end);
    merge(array, begin, mid, end);
}
  
// Function to print an array
void printArray(int A[], int size)
{
    for (auto i = 0; i < size; i++)
        cout << A[i] << " ";
}

/*------------------------------------------------------------------------------
                        Complete Graph Uniform between [0,1] 
------------------------------------------------------------------------------*/

vector<edge> make_oneD_graph(int n)
{
    vector<edge> vect;
    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            edge edge_new = make_tuple((float) rand()/RAND_MAX, i, j);
            vect.push_back(edge_new);
        }
    }
    return vect;
}

/*------------------------------------------------------------------------------
                    Complete Graph Uniform inside Unit Square 
------------------------------------------------------------------------------*/

double** make_twoD_graph(int n)
{
    double** arr = new double*[n];
    for (int i = 0; i < n; ++i) {
        arr[i] = new double[n];
        for (int j = 0; j < n; ++j) {
            arr[i][j] = (((double)rand()/(double)RAND_MAX), 
                         ((double)rand()/(double)RAND_MAX));;
        }
    }
    return arr;
}

/*------------------------------------------------------------------------------
                    Complete Graph Uniform inside Unit Cube
------------------------------------------------------------------------------*/

double** make_threeD_graph(int n)
{
    double** arr = new double*[n];
    for (int i = 0; i < n; ++i) {
        arr[i] = new double[n];
        for (int j = 0; j < n; ++j) {
            arr[i][j] = (((double)rand()/(double)RAND_MAX), 
                         ((double)rand()/(double)RAND_MAX));;
        }
    }
    return arr;
}

/*------------------------------------------------------------------------------
                                 Kruskal's Algorithm
------------------------------------------------------------------------------*/

double kruskal(vector<edge>& edges, int n) {
    MergeSort(edges, 0, n); // sort edges from least to greatest weight

    set<int> s = {}; // set of vertices in the MST
    int sum = 0; // accumulator for sum of weigths in the MST
    int index = 0; // keep track of edges already visited
    
    // keep adding edges until all vertices are in the MST
    while(s.size() < n) {
        bool add_edge = false;
        int i = get<1>(edges[index]);
        int j = get<2>(edges[index]);
        if(s.find(i) != s.end()) {
            add_edge = true;
            s.insert(i);
        }
        if(s.find(j) != s.end()) {
            add_edge = true;
            s.insert(j);
        }
        if(add_edge) {
            sum += get<0>(edges[index]);
        }
        index++;
    }
    return sum;
}

/*------------------------------------------------------------------------------
                            Testing Pruning Upper Bound
------------------------------------------------------------------------------*/

// Insert here

/*******************************************************************************
                                    Main 
*******************************************************************************/
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

