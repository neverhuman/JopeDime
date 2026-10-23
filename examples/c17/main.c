#include "model.h"
#include <inttypes.h>
#include <stdio.h>
int main(void) {
    double x[]={0,1,1,2,2,3,3,5,4,8,5,13,6,21,7,34};
    double y[]={0,1,2,3,4,5,6,7}, test[]={2,3,5,13}, predictions[2];
    uint64_t handle=0;
    int32_t status=jope_model_fit(x,16,y,8,8,2,&handle);
    if(status) {fprintf(stderr,"fit: %" PRId32 "\n",status); return 1;}
    status=jope_model_predict(handle,test,4,2,2,predictions,2);
    if(status) {fprintf(stderr,"predict: %" PRId32 "\n",status); (void)jope_model_free(handle); return 2;}
    for(size_t i=0;i<2;i++) printf("%.17g\n",predictions[i]);
    return jope_model_free(handle)==JOPE_MODEL_OK ? 0 : 3;
}
