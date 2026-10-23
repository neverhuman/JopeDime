#include <cstddef>
#include <cstdint>
#include <cassert>
#include <limits>
#include <iostream>
extern "C" {
int __PREFIX__fit(const double*,std::size_t,const double*,std::size_t,std::size_t,std::size_t,std::uint64_t*);
int __PREFIX__free(std::uint64_t);
int __PREFIX__open(std::uint64_t*);
int __PREFIX__linear_width(std::uint64_t,std::size_t*);
int __PREFIX__linear(std::uint64_t,const double*,std::size_t,std::size_t,std::size_t,double*,std::size_t,double*,std::size_t);
int __PREFIX__text(std::uint64_t,const unsigned char*,std::size_t,double*,std::size_t);
}
int main(){
 double x[]={0,1,1,2,2,3,3,1},y[]={0,1,2,4};std::uint64_t h=0,out_handle=999;
 assert(__PREFIX__open(&out_handle)==5 && out_handle==999);
 assert(__PREFIX__open(nullptr)==1);
 assert(__PREFIX__fit(x,8,y,4,4,2,&h)==0);
 std::size_t width=777;
 assert(__PREFIX__linear_width(0,&width)==2 && width==777);
 assert(__PREFIX__linear_width(h,nullptr)==1);
 assert(__PREFIX__linear_width(h,&width)==0 && width==2);
 double features[]={-99,-99},weights[]={-88,-88,-88};
 const auto untouched=[&]{return features[0]==-99 && features[1]==-99 && weights[0]==-88 && weights[1]==-88 && weights[2]==-88;};
 assert(__PREFIX__linear(h,x,2,1,2,features,1,weights,3)==1 && untouched());
 assert(__PREFIX__linear(h,x,2,1,2,features,2,weights,2)==1 && untouched());
 assert(__PREFIX__linear(h,nullptr,2,1,2,features,2,weights,3)==1 && untouched());
 assert(__PREFIX__linear(h,x,2,1,2,features,2,features,3)==1 && untouched());
 assert(__PREFIX__linear(h,x,2,1,2,nullptr,2,weights,3)==1 && untouched());
 assert(__PREFIX__linear(h,x,2,std::numeric_limits<std::size_t>::max(),2,features,2,weights,3)==1 && untouched());
 alignas(double) unsigned char misaligned[24]{};
 assert(__PREFIX__linear(h,reinterpret_cast<double*>(misaligned+1),2,1,2,features,2,weights,3)==1 && untouched());
 const unsigned char invalid[]={0xff};double out=-7;
 assert(__PREFIX__text(h,invalid,1,&out,1)==1 && out==-7);
 const unsigned char valid[]="a,b\n1,2\n";
 assert(__PREFIX__text(h,valid,sizeof(valid)-1,&out,1)==5 && out==-7);
 assert(__PREFIX__linear(h,x,2,1,2,features,2,weights,3)==0);
 // Inputs may alias output: all foreign input is copied before output commits.
 assert(__PREFIX__linear(h,x,2,1,2,x,2,weights,3)==0);
 assert(__PREFIX__free(h)==0);
 width=777;assert(__PREFIX__linear_width(h,&width)==2 && width==777);
 assert(__PREFIX__free(h)==2);
 std::cout<<"CUDA native boundary passed\n";
}
