#include <iostream>
#include <iomanip>


#include <flann/flann.hpp>
#include <flann/io/hdf5.h>

#include <stdio.h>

using namespace std;

int main(int argc, const char* argv[])
{
int nn = 3;

flann::Matrix<double> dataset;
flann::Matrix<double> query;

flann::load_from_file(dataset, "/home/travis/downloads/dataset.hdf5","dataset");
flann::load_from_file(query, "/home/travis/downloads/dataset.hdf5","query");

flann::Matrix<int> indices(new int[query.rows*nn], query.rows, nn);
flann::Matrix<double> dists(new double[query.rows*nn], query.rows, nn);

// construct an randomized kd-tree index using 4 kd-trees
std::cout << "Building randomized kd-tree indices..." << endl;
flann::Index<flann::L2<double> > index(dataset, flann::AutotunedIndexParams(
                                                               0.9,
                                                               0.01,
                                                               0,
                                                               0.1 ));
index.buildIndex();

// do a knn search, using 128 checks
std::cout << "Searching query using 128 checks" << endl;
index.knnSearch(query, indices, dists, nn, flann::SearchParams(128));
flann::save_to_file(indices,"result.hdf5","result");

  delete[] dataset.ptr();
  delete[] query.ptr();
  delete[] indices.ptr();
  delete[] dists.ptr();

  return 0;
}