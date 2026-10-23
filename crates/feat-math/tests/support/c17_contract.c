#include "model.h"
#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
_Static_assert(__STDC_VERSION__ >= 201710L, "C17 required for this contract");

static double x[] = {0,1, 1,2, 2,3, 3,5, 4,8, 5,13, 6,21, 7,34};
static double y[] = {0,1,2,3,4,5,6,7};
static double test[] = {2,3, 5,13};
static uint64_t shared;
static double expected[2];
static void *predict_thread(void *arg) {
    (void)arg;
    for (int i=0;i<20;i++) {
        double out[2];
        assert(jope_model_predict(shared,test,4,2,2,out,2)==JOPE_MODEL_OK);
        assert(memcmp(out,expected,sizeof(out))==0);
    }
    return NULL;
}
int main(void) {
    uint64_t h=UINT64_C(999), h2=0;
    double out[4]={-99,-99,-99,-99};
    assert(jope_model_abi_version()==1);
    assert(jope_model_fit(NULL,16,y,8,8,2,&h)==JOPE_MODEL_INVALID_ARGUMENT);
    assert(h==999);
    assert(jope_model_fit(x,15,y,8,8,2,&h)==JOPE_MODEL_INVALID_ARGUMENT);
    assert(jope_model_fit(x,16,NULL,8,8,2,&h)==JOPE_MODEL_INVALID_ARGUMENT);
    assert(jope_model_fit(x,16,y,7,8,2,&h)==JOPE_MODEL_INVALID_ARGUMENT);
    assert(jope_model_fit(x,16,y,8,8,2,NULL)==JOPE_MODEL_INVALID_ARGUMENT);
    assert(jope_model_fit(x,0,y,0,0,2,&h)==JOPE_MODEL_INVALID_ARGUMENT);
    assert(jope_model_fit(x,0,y,8,8,0,&h)==JOPE_MODEL_INVALID_ARGUMENT);
    assert(jope_model_fit(x,SIZE_MAX,y,8,SIZE_MAX,2,&h)==JOPE_MODEL_INVALID_ARGUMENT);
    assert(jope_model_fit((const double *)((const char*)x+1),16,y,8,8,2,&h)==JOPE_MODEL_INVALID_ARGUMENT);
    assert(jope_model_fit(x,16,y,8,8,2,&h)==0);
    assert(h!=0 && h!=999);
    assert(jope_model_fit(x,16,y,8,8,2,&h2)==0 && h2!=h);
    assert(jope_model_predict(h,test,4,2,2,out,1)==JOPE_MODEL_INVALID_ARGUMENT);
    assert(jope_model_predict(h,test,3,2,2,out,2)==JOPE_MODEL_INVALID_ARGUMENT);
    assert(jope_model_predict(h,test,4,1,4,out,2)==JOPE_MODEL_INVALID_ARGUMENT);
    assert(jope_model_predict(h,test,4,2,2,NULL,2)==JOPE_MODEL_INVALID_ARGUMENT);
    assert(jope_model_predict(0,test,4,2,2,out,2)==JOPE_MODEL_INVALID_HANDLE);
    assert(jope_model_predict(UINT64_MAX,test,4,2,2,out,2)==JOPE_MODEL_INVALID_HANDLE);
    assert(out[0]==-99 && out[1]==-99);
    assert(jope_model_predict(h,NULL,0,0,2,NULL,0)==0);
    assert(jope_model_predict(h,test,4,2,2,out,4)==0);
    assert(out[2]==-99 && out[3]==-99);
    assert(jope_model_predict(h2,test,4,2,2,expected,2)==0);
    assert(memcmp(expected,out,sizeof(expected))==0);
    /* Copy-before-write permits the caller's output to alias its input. */
    double alias[4]; memcpy(alias,test,sizeof(alias));
    assert(jope_model_predict(h,alias,4,2,2,alias,2)==0);
    assert(memcmp(alias,expected,sizeof(expected))==0);
    shared=h; pthread_t threads[4];
    for(int i=0;i<4;i++) assert(pthread_create(&threads[i],NULL,predict_thread,NULL)==0);
    for(int i=0;i<4;i++) assert(pthread_join(threads[i],NULL)==0);
    double score=-99;
    assert(jope_model_score(h,x,16,y,8,8,2,&score)==0);
    assert(score>0.99 && score<=1.0);
    score=-99;
    assert(jope_model_score(h,x,16,y,7,8,2,&score)==JOPE_MODEL_INVALID_ARGUMENT);
    assert(score==-99);
    assert(jope_model_score(h,x,16,y,8,8,2,NULL)==JOPE_MODEL_INVALID_ARGUMENT);
    assert(jope_model_score(0,x,16,y,8,8,2,&score)==JOPE_MODEL_INVALID_HANDLE);
    assert(jope_model_free(h)==0);
    assert(jope_model_free(h)==JOPE_MODEL_INVALID_HANDLE);
    assert(jope_model_free(0)==JOPE_MODEL_INVALID_HANDLE);
    assert(jope_model_predict(h,test,4,2,2,out,2)==JOPE_MODEL_INVALID_HANDLE);
    assert(jope_model_fit(x,16,y,8,8,2,&shared)==0 && shared!=h);
    assert(jope_model_free(shared)==0);
    assert(jope_model_predict(h2,test,4,2,2,out,2)==0);
    assert(jope_model_free(h2)==0);
    puts("C17 ABI contracts passed");
    return 0;
}
