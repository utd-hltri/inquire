#include <flann/flann.hpp>
#include "edu_utdallas_hltri_inquire_ANN.h"

flann::Index<flann::L2<double> >* _index;

/*
 * Class:     edu_utdallas_hltri_inquire_ANN
 * Method:    findKNN
 * Signature: ([DI[I[D)V
 */
JNIEXPORT void JNICALL Java_edu_utdallas_hltri_inquire_ANN_findKNN
  (JNIEnv * env, jclass clazz, jdoubleArray jQuery, jint checks, jintArray jIndices, jdoubleArray jWeights)
{
  long    N     = env->GetArrayLength(jQuery);
  long    nn    = env->GetArrayLength(jIndices);
  jdouble* qArray = env->GetDoubleArrayElements(jQuery, 0);

  flann::Matrix<double>   query(qArray, 1, N);
  flann::Matrix<int>    indices(new int[ query.rows * nn ], query.rows, nn);
  flann::Matrix<double>   dists(new double[ query.rows * nn ], query.rows, nn);

  _index->knnSearch(query, indices, dists, nn, flann::SearchParams(checks));

  env->SetIntArrayRegion(jIndices, 0, nn, indices[0]);
  env->SetDoubleArrayRegion(jWeights, 0, nn, dists[0]);

  delete[] query.ptr();
  delete[] indices.ptr();
  delete[] dists.ptr();
}

/*
 * Class:     edu_utdallas_hltri_inquire_ANN
 * Method:    saveIndex
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_edu_utdallas_hltri_inquire_ANN_saveIndex
  (JNIEnv * env, jclass clazz, jstring filename)
{
  const char *str= env->GetStringUTFChars(filename, 0);
  std::string filestr(str);
  _index->save(str);
}

/*
 * Class:     edu_utdallas_hltri_inquire_ANN
 * Method:    loadIndex
 * Signature: ([DIILjava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_edu_utdallas_hltri_inquire_ANN_loadIndex
  (JNIEnv * env, jclass clazz, jdoubleArray jData, jint rows, jint cols, jstring filename)
{
  long    N    = env->GetArrayLength(jData);           // Access the array length
  jdouble* data = env->GetDoubleArrayElements(jData, 0);
  flann::Matrix<double> dataset(data, rows, cols);

  const char *str= env->GetStringUTFChars(filename, 0);
  std::string filestr(str);

  _index = new flann::Index<flann::L2<double> >(dataset, flann::SavedIndexParams(filestr));
  _index->buildIndex();
}

/*
 * Class:     edu_utdallas_hltri_inquire_ANN
 * Method:    buildLinearIndex
 * Signature: ([DII)V
 */
JNIEXPORT void JNICALL Java_edu_utdallas_hltri_inquire_ANN_buildLinearIndex
  (JNIEnv * env, jclass clazz, jdoubleArray jData, jint rows, jint cols)
{
 long    N    = env->GetArrayLength(jData);           // Access the array length
 jdouble* data = env->GetDoubleArrayElements(jData, 0);
 flann::Matrix<double> dataset(data, rows, cols);

 _index = new flann::Index<flann::L2<double> >(dataset, flann::LinearIndexParams());

 _index->buildIndex();
}

/*
 * Class:     edu_utdallas_hltri_inquire_ANN
 * Method:    buildKDTreeIndex
 * Signature: ([DIII)V
 */
JNIEXPORT void JNICALL Java_edu_utdallas_hltri_inquire_ANN_buildKDTreeIndex
  (JNIEnv * env, jclass clazz, jdoubleArray jData, jint rows, jint cols, jint trees)
{
  long    N    = env->GetArrayLength(jData);           // Access the array length
  jdouble* data = env->GetDoubleArrayElements(jData, 0);
  flann::Matrix<double> dataset(data, rows, cols);

  _index = new flann::Index<flann::L2<double> >(dataset, flann::KDTreeIndexParams(
    trees
  ));

  _index->buildIndex();
}

/*
 * Class:     edu_utdallas_hltri_inquire_ANN
 * Method:    buildKMeansIndex
 * Signature: ([DIIIID)V
 */
JNIEXPORT void JNICALL Java_edu_utdallas_hltri_inquire_ANN_buildKMeansIndex
  (JNIEnv * env, jclass clazz, jdoubleArray jData, jint rows, jint cols, jint branching, jint iterations, jdouble cb_index)
{
  long    N    = env->GetArrayLength(jData);           // Access the array length
  jdouble* data = env->GetDoubleArrayElements(jData, 0);
  flann::Matrix<double> dataset(data, rows, cols);

  _index = new flann::Index<flann::L2<double> >(dataset, flann::KMeansIndexParams(
    branching,
    iterations,
    flann::FLANN_CENTERS_RANDOM,
    (float) cb_index
  ));

  _index->buildIndex();
}

/*
 * Class:     edu_utdallas_hltri_inquire_ANN
 * Method:    buildCompositeIndex
 * Signature: ([DIIIIID)V
 */
JNIEXPORT void JNICALL Java_edu_utdallas_hltri_inquire_ANN_buildCompositeIndex
  (JNIEnv * env, jclass clazz, jdoubleArray jData, jint rows, jint cols, jint trees, jint branching, jint iterations, jdouble cb_index)
{
  long    N    = env->GetArrayLength(jData);           // Access the array length
  jdouble* data = env->GetDoubleArrayElements(jData, 0);
  flann::Matrix<double> dataset(data, rows, cols);

  _index = new flann::Index<flann::L2<double> >(dataset, flann::CompositeIndexParams(
    trees,
    branching,
    iterations,
    flann::FLANN_CENTERS_RANDOM,
    (float) cb_index
  ));

  _index->buildIndex();
}

/*
 * Class:     edu_utdallas_hltri_inquire_ANN
 * Method:    buildLSHIndex
 * Signature: ([DIIIII)V
 */
JNIEXPORT void JNICALL Java_edu_utdallas_hltri_inquire_ANN_buildLSHIndex
  (JNIEnv * env, jclass clazz, jdoubleArray jData, jint rows, jint cols, jint table_number, jint key_size, jint multi_probe_level)
{
  long    N    = env->GetArrayLength(jData);           // Access the array length
  jdouble* data = env->GetDoubleArrayElements(jData, 0);
  flann::Matrix<double> dataset(data, rows, cols);

  _index = new flann::Index<flann::L2<double> >(dataset, flann::LshIndexParams(
    (unsigned int) table_number,
    (unsigned int) key_size,
    (unsigned int) multi_probe_level
  ));

  _index->buildIndex();
}

/*
 * Class:     edu_utdallas_hltri_inquire_ANN
 * Method:    buildAutoIndex
 * Signature: ([DIIDDDD)V
 */
JNIEXPORT void JNICALL Java_edu_utdallas_hltri_inquire_ANN_buildAutoIndex
  (JNIEnv * env, jclass clazz, jdoubleArray jData, jint rows, jint cols, jdouble target_precision, jdouble build_weight, jdouble memory_weight, jdouble sample_fraction)
{
  long    N    = env->GetArrayLength(jData);           // Access the array length
  jdouble* data = env->GetDoubleArrayElements(jData, 0);
  flann::Matrix<double> dataset(data, rows, cols);

  _index = new flann::Index<flann::L2<double> >(dataset, flann::AutotunedIndexParams(
    (float) target_precision,
    (float) build_weight,
    (float) memory_weight,
    (float) sample_fraction
  ));

  _index->buildIndex();
}


