/*******************************************************************************
                CS124 Programming Set 1 - Elizabeth Zhong, Helen Xiao 
*******************************************************************************/
#include <iostream>
using namespace std;
#include <stdlib.h>
#include <set>
#include<vector>
#include <math.h> 
using namespace std::chrono;
#include <unordered_set>

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
                              Disjoint Set Class
------------------------------------------------------------------------------*/

class Disjoint {
    public:
        vector<int> parent;
        vector<int> rank;
    
    // constructor
    Disjoint(int n){
        parent = vector<int>(n);
        rank = vector<int>(n);
    }

    // makeset
    void makeset(int x) {
        parent[x] = x;
        rank[x] = 0;
    }

    // find
    int find(int x) {
        while (x != parent[x]){
            x = parent[x];
        }
        return x;
    }

    // union
    void union_d(int x, int y) {
        int r_x = find(x);
        int r_y = find(y);
        if (r_x == r_y) {
            return;
        }
        if(rank[r_x] > rank[r_y]) {
            parent[r_y] = r_x;
        } else {
            parent[r_x] = r_y;
            if(rank[r_x] == rank[r_y]) {
                rank[r_y] += 1;
            }
        }
    }
};

/*------------------------------------------------------------------------------
                               Kruskal's Algorithm
------------------------------------------------------------------------------*/

tuple<double, double> kruskal(vector<edge>& edges, int n) {
    // create disjoint set
    Disjoint Set = Disjoint(n);
    for (int i = 0; i < n; i++) {
        Set.makeset(i);
    }

    // keep track of weighted edge sum and max edge
    double sum = 0;
    double max_edge = 0;

    // sort edges from least to greatest weight
    MergeSort(edges, 0, edges.size()-1); 

    // loop through edges to add to MST
    for(int i = 0; i < edges.size(); i++) {
        int a = get<1>(edges[i]);
        int b = get<2>(edges[i]);
        if(Set.find(a) != Set.find(b)) {
            double weight = get<0>(edges[i]);
            sum += weight;
            Set.union_d(a, b);
            if(weight > max_edge) {
                max_edge = weight;
            }
        }
    }

    // returns a tuple of the edge weight sum and max edge (for pruning)
    return make_tuple(sum, max_edge);
};


/*------------------------------------------------------------------------------
                        Complete Graph Uniform between [0,1] 
------------------------------------------------------------------------------*/

// create a list of edges for 1D graph (pruning along the way)
vector<edge> make_oneD_graph(int n)
{
    // pruning threshold
    double threshold = 3.156 * pow((double)n, -0.847);
    vector<edge> vect;
    for (int i = 0; i < n; ++i) {
        for (int j = i+1; j < n; ++j) {
            double weight = (float) rand()/RAND_MAX;
            if (weight < threshold) {
                edge edge_new = make_tuple(weight, i, j);
                vect.push_back(edge_new);
            }
        }
    }
    return vect;
}


/*------------------------------------------------------------------------------
                    Complete Graph Uniform inside Unit Square 
------------------------------------------------------------------------------*/

// calculate distance between 2 points in 2D
double distanceCalculate(tuple<double,double> n1, tuple<double,double>n2)
{
	double x = get<0>(n1) - get<0>(n2); 
	double y = get<1>(n1) - get<1>(n2);
	double dist;

	dist = pow(x, 2) + pow(y, 2);       
	dist = sqrt(dist);                  

	return dist;
}

// create a list of edges for 2D graph (pruning along the way)
vector<edge> make_twoD_graph(int n)
{
    // pruning threshold
    double threshold = 1.776 * pow((double) n, -0.478); 

    vector<edge> vect;
    tuple<double, double> arr[n];
    for (int k = 0; k < n; ++k) {
        arr[k] = make_tuple((double) rand()/RAND_MAX, (double) rand()/RAND_MAX);
    }

    for (int i = 1; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            double weight = distanceCalculate(arr[i], arr[j]);
            if (weight < threshold) {
                edge edge_new = make_tuple(weight, i, j);
                vect.push_back(edge_new);
            }
        }
    }
    return vect;
}


/*------------------------------------------------------------------------------
                    Complete Graph Uniform inside Unit Cube
------------------------------------------------------------------------------*/

// calculate distance between 2 points in 3D
double threeDistanceCalculate(tuple<double,double,double> n1, tuple<double,double,double>n2)
{
	double x = get<0>(n1) - get<0>(n2); 
	double y = get<1>(n1) - get<1>(n2);
    double z = get<2>(n1) - get<2>(n2);
	double dist;

	dist = pow(x, 2) + pow(y, 2) + pow(z, 2);       
	dist = sqrt(dist);                  

	return dist;
}

// create a list of edges for 3D graph (pruning along the way)
vector<edge> make_threeD_graph(int n)
{
    vector<edge> vect;
    tuple<double, double, double> arr[n];
    for (int k = 0; k < n; ++k) {
        arr[k] = make_tuple((double) rand()/RAND_MAX, 
                            (double) rand()/RAND_MAX, 
                            (double) rand()/RAND_MAX);
    }

    // pruning threshold
    double threshold = 1.644 * pow((double)n, -0.327); 
    for (int i = 0; i < n; ++i) {
        for (int j = i+1; j < n; ++j) {
            double weight = threeDistanceCalculate(arr[i], arr[j]);
            if(weight < threshold) {
                edge edge_new = make_tuple(threeDistanceCalculate(arr[i], arr[j]), i, j);
                vect.push_back(edge_new);
            }
        }
    }
    return vect;
}

/*------------------------------------------------------------------------------
                    Complete Graph Uniform inside Hyper Cube
------------------------------------------------------------------------------*/

// calculate distance between 2 points in 4D
double fourDistanceCalculate(tuple<double,double,double,double> n1, tuple<double,double,double,double>n2)
{
    //calculating number to square in next step
	double x = get<0>(n1) - get<0>(n2); 
	double y = get<1>(n1) - get<1>(n2);
    double w = get<2>(n1) - get<2>(n2);
    double z = get<3>(n1) - get<3>(n2);
	double dist;

    //calculating Euclidean distance
	dist = pow(x, 2) + pow(y, 2) + pow(w, 2) + pow(z, 2);       
	dist = sqrt(dist);                  

	return dist;
}

// create a list of edges for 4D graph (pruning along the way)
vector<edge> make_fourD_graph(int n)
{
    vector<edge> vect;
    tuple<double, double, double, double> arr[n];
    for (int k = 0; k < n; ++k) {
        arr[k] = make_tuple((double) rand()/RAND_MAX, 
                            (double) rand()/RAND_MAX, 
                            (double) rand()/RAND_MAX, 
                            (double) rand()/RAND_MAX);
    }

    // pruning threshold
    double threshold = 1.524 * pow((double)n, -0.458);
    for (int i = 0; i < n; ++i) {
        for (int j = i+1; j < n; ++j) {
            double weight = fourDistanceCalculate(arr[i], arr[j]);
            if(weight < threshold) {
                edge edge_new = make_tuple(fourDistanceCalculate(arr[i], arr[j]), i, j);
                vect.push_back(edge_new);
            }
        }
    }
    return vect;
}

/*------------------------------------------------------------------------------
                            Testing Pruning Upper Bound
------------------------------------------------------------------------------*/

// Testing the maximum weighted edge of a graph
void testing_max() {
    double tpn = 5.; //trials per n value
    int min_pow = 7; // minimum power of 2 we test
    int cap_pow = 12; // maximum power of 2 we test
    
    // run trials
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


/*------------------------------------------------------------------------------
                                    Run Trials
------------------------------------------------------------------------------*/

// Run trials for n = 2^j where 7 <= j <= 18
void run_trials() {
    double tpn = 5.; //trials per n value
    int min_pow = 18; // minimum power of 2 we test
    int cap_pow = 18; // maximum power of 2 we test

    // run trials
    for(int exp = min_pow; exp <= cap_pow; exp++) {
        double avg_sum = 0;
        double n = pow(2., exp);
        double av_duration = 0.;
        fprintf(stdout, "Start %f\n", n);
        for(int i = 0; i < tpn; i++) {
            auto start = high_resolution_clock::now();
            vector<edge> v = make_fourD_graph(n);
            avg_sum += get<0>(kruskal(v, n));
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<milliseconds>(stop - start);
            av_duration += duration.count();
        }
        av_duration /= tpn;
        avg_sum /= tpn;
        fprintf(stdout, "n: %f, avg_sum: %f, av_duration: %f\n",  
                n, avg_sum, av_duration);
    }
}

/*------------------------------------------------------------------------------
                                Run Specified Trial
------------------------------------------------------------------------------*/

void run(int n, int trials, int dim) {
    fprintf(stdout, "n = %i, trials = %i, dim = %i\n", n, trials, dim);
    double avg_sum = 0;
    for(int i = 0; i < trials; i++) {
        vector<edge> v;
        if(dim == 1) {
            v = make_oneD_graph(n);
        } else if(dim == 2) {
            v = make_twoD_graph(n);
        } else if(dim == 3) {
            v = make_threeD_graph(n);
        } else if(dim == 4) {
            v = make_fourD_graph(n);
        }
        avg_sum += get<0>(kruskal(v, n));
    }
    avg_sum /= trials;
    fprintf(stdout, "Average Tree Size: %f\n", avg_sum);
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

    run(n, trials, dim);
}