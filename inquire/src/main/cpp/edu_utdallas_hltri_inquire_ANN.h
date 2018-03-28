/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class edu_utdallas_hltri_inquire_ANN */

#ifndef _Included_edu_utdallas_hltri_inquire_ANN
#define _Included_edu_utdallas_hltri_inquire_ANN
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     edu_utdallas_hltri_inquire_ANN
 * Method:    findKNN
 * Signature: ([DI[I[D)V
 */
JNIEXPORT void JNICALL Java_edu_utdallas_hltri_inquire_ANN_findKNN
  (JNIEnv *, jclass, jdoubleArray, jint, jintArray, jdoubleArray);

/*
 * Class:     edu_utdallas_hltri_inquire_ANN
 * Method:    saveIndex
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_edu_utdallas_hltri_inquire_ANN_saveIndex
  (JNIEnv *, jclass, jstring);

/*
 * Class:     edu_utdallas_hltri_inquire_ANN
 * Method:    loadIndex
 * Signature: ([DIILjava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_edu_utdallas_hltri_inquire_ANN_loadIndex
  (JNIEnv *, jclass, jdoubleArray, jint, jint, jstring);

/*
 * Class:     edu_utdallas_hltri_inquire_ANN
 * Method:    buildLinearIndex
 * Signature: ([DII)V
 */
JNIEXPORT void JNICALL Java_edu_utdallas_hltri_inquire_ANN_buildLinearIndex
  (JNIEnv *, jclass, jdoubleArray, jint, jint);

/*
 * Class:     edu_utdallas_hltri_inquire_ANN
 * Method:    buildKDTreeIndex
 * Signature: ([DIII)V
 */
JNIEXPORT void JNICALL Java_edu_utdallas_hltri_inquire_ANN_buildKDTreeIndex
  (JNIEnv *, jclass, jdoubleArray, jint, jint, jint);

/*
 * Class:     edu_utdallas_hltri_inquire_ANN
 * Method:    buildKMeansIndex
 * Signature: ([DIIIID)V
 */
JNIEXPORT void JNICALL Java_edu_utdallas_hltri_inquire_ANN_buildKMeansIndex
  (JNIEnv *, jclass, jdoubleArray, jint, jint, jint, jint, jdouble);

/*
 * Class:     edu_utdallas_hltri_inquire_ANN
 * Method:    buildCompositeIndex
 * Signature: ([DIIIIID)V
 */
JNIEXPORT void JNICALL Java_edu_utdallas_hltri_inquire_ANN_buildCompositeIndex
  (JNIEnv *, jclass, jdoubleArray, jint, jint, jint, jint, jint, jdouble);

/*
 * Class:     edu_utdallas_hltri_inquire_ANN
 * Method:    buildLSHIndex
 * Signature: ([DIIIII)V
 */
JNIEXPORT void JNICALL Java_edu_utdallas_hltri_inquire_ANN_buildLSHIndex
  (JNIEnv *, jclass, jdoubleArray, jint, jint, jint, jint, jint);

/*
 * Class:     edu_utdallas_hltri_inquire_ANN
 * Method:    buildAutoIndex
 * Signature: ([DIIDDDD)V
 */
JNIEXPORT void JNICALL Java_edu_utdallas_hltri_inquire_ANN_buildAutoIndex
  (JNIEnv *, jclass, jdoubleArray, jint, jint, jdouble, jdouble, jdouble, jdouble);

#ifdef __cplusplus
}
#endif
#endif