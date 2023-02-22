/*******************************************************************************
                CS124 Programming Set 1 - Elizabeth Zhong, Helen Xiao 
*******************************************************************************/
#include <iostream>
using namespace std;
#include <stdlib.h>
#include <set>
#include<vector>
#include <math.h> 

// define new type, "edge" (edge weight, fst endpoint, snd endpoint)
typedef std::tuple<double, int, int> edge;

void print_v(vector<edge> v) {
    fprintf(stdout, "New Edge Vector\n");
    for(int i=0; i < v.size(); i++) {
        double weight = get<0>(v[i]);
        int x = get<1>(v[i]);
        int y = get<2>(v[i]);
        fprintf(stdout, "(weight: %f, x: %i, y: %i) \n", weight, x, y);
    }
}


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
                        Complete Graph Uniform between [0,1] 
------------------------------------------------------------------------------*/

vector<edge> make_oneD_graph(int n)
{
    vector<edge> vect;
    for (int i = 0; i < n; ++i) {
        for (int j = i+1; j < n; ++j) {
            edge edge_new = make_tuple((float) rand()/RAND_MAX, i, j);
            vect.push_back(edge_new);
        }
    }
    return vect;
}

/*
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
}*/

/*------------------------------------------------------------------------------
                    Complete Graph Uniform inside Unit Square 
------------------------------------------------------------------------------*/

double distanceCalculate(tuple<double,double> n1, tuple<double,double>n2)
{
	double x = get<0>(n1) - get<0>(n2); 
	double y = get<1>(n1) - get<1>(n2);
	double dist;

	dist = pow(x, 2) + pow(y, 2);       
	dist = sqrt(dist);                  

	return dist;
}

vector<edge> make_twoD_graph(int n)
{
    vector<edge> vect;
    tuple<double, double> arr[n];
    for (int k = 0; k < n; ++k) {
        arr[k] = make_tuple((double) rand()/RAND_MAX, (double) rand()/RAND_MAX);
    }

    for (int i = 1; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            edge edge_new = make_tuple(distanceCalculate(arr[i], arr[j]), i, j);
            vect.push_back(edge_new);
        }
    }
    return vect;
}

/*------------------------------------------------------------------------------
                    Complete Graph Uniform inside Unit Cube
------------------------------------------------------------------------------*/

double distanceCalculate(tuple<double,double,double> n1, tuple<double,double,double>n2)
{
	double x = get<0>(n1) - get<0>(n2); 
	double y = get<1>(n1) - get<1>(n2);
	double dist;

	dist = pow(x, 2) + pow(y, 2);       
	dist = sqrt(dist);                  

	return dist;
}

vector<edge> make_threeD_graph(int n)
{
    vector<edge> vect;
    tuple<double, double, double> arr[n];
    double threshold = 2.8 * pow((double)n, -0.473);
    for (int k = 0; k < n; ++k) {
        arr[k] = make_tuple((double) rand()/RAND_MAX, (double) rand()/RAND_MAX, 
                            (double) rand()/RAND_MAX);
    }

    for (int i = 1; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            double weight = distanceCalculate(arr[i], arr[j]);
            if(weight < threshold) {
                edge edge_new = make_tuple(distanceCalculate(arr[i], arr[j]), i, j);
                vect.push_back(edge_new);
            }
        }
    }
    return vect;
}

/*******************************************************************************
                    Complete Graph Uniform inside Hyper Cube
*******************************************************************************/

double distanceCalculate(tuple<double,double,double,double> n1, tuple<double,double,double,double>n2)
{
	double x = get<0>(n1) - get<0>(n2); //calculating number to square in next step
	double y = get<1>(n1) - get<1>(n2);
	double dist;

	dist = pow(x, 2) + pow(y, 2);       //calculating Euclidean distance
	dist = sqrt(dist);                  

	return dist;
}

vector<edge> make_fourD_graph(int n)
{
    double threshold = 2.54 * pow((double)n, -0.458);
    vector<edge> vect;
    tuple<double, double, double,double> arr[n];
    for (int k = 0; k < n; ++k) {
        arr[k] = make_tuple((double) rand()/RAND_MAX, (double) rand()/RAND_MAX, 
                            (double) rand()/RAND_MAX, (double) rand()/RAND_MAX);
    }

    for (int i = 1; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            double weight = distanceCalculate(arr[i], arr[j]);
            if(weight < threshold) {
                edge edge_new = make_tuple(distanceCalculate(arr[i], arr[j]), i, j);
                vect.push_back(edge_new);
            }
        }
    }
    return vect;
}

/*------------------------------------------------------------------------------
                                 Kruskal's Algorithm
------------------------------------------------------------------------------*/

tuple<double, double> kruskal(vector<edge>& edges, int n) {
    MergeSort(edges, 0, edges.size()-1); // sort edges from least to greatest weight

    set<int> s = {}; // set of vertices in the MST
    double sum = 0.; // accumulator for sum of weigths in the MST
    int index = 0; // keep track of edges already visited
    double max_edge = 0.;
    
    // keep adding edges until all vertices are in the MST
    while(s.size() < n && index < edges.size()) {
        bool add_edge = false;
        int i = get<1>(edges[index]);
        int j = get<2>(edges[index]);

        // add endpoint if they are not in s
        if(s.find(i) == s.end()) {
            add_edge = true;
            s.insert(i);
        }
        if(s.find(j) == s.end()) {
            add_edge = true;
            s.insert(j);
        }

        // add edge weight accordingly
        if(add_edge == true) {
            double curr_edge_weight = get<0>(edges[index]);
            sum += get<0>(edges[index]);
            if(curr_edge_weight > max_edge) {
                max_edge = curr_edge_weight;
            }
        }
        index++;
    }
    return make_tuple(sum, max_edge);
}

/*------------------------------------------------------------------------------
                            Testing Pruning Upper Bound
------------------------------------------------------------------------------*/

double tpn = 5.; //trials per n value
int min_pow = 7;
int cap_pow = 13; // maximum power of 2 we test

void testing_max() {
    for(int exp = min_pow; exp <= cap_pow; exp++) {
        double avg_max = 0;
        double n = pow(2., exp);
        for(int i = 0; i < tpn; i++) {
            vector<edge> v = make_fourD_graph(n);
            avg_max += get<1>(kruskal(v, n));
        }
        avg_max /= tpn;
        fprintf(stdout, "n: %f, avg_max: %f\n", n, avg_max);
    }
}

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


    fprintf(stdout, "n: %i trials: %i dim: %i\n", n, trials, dim);

    /*
    for(int i = 1; i < 15; i++) {
        vector<edge> v = make_oneD_graph(pow(2, i));
        fprintf(stdout, "n: 2^%i, max weight: %f\n", i, get<1>(kruskal(v, 128)));
    } */

    testing_max();
    
}

