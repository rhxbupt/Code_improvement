The Readme file of the project1
author:Haoxu Ren
================================================================
The introduction of this project: 

There is a folder named sm-centric.

in this folder, there are a c++ file named SM-centric_transformation.cpp. This code implement the source to source transformation from CUDA code to 
SM-Centric format code. And there also a txt file named CMakeLists.txt, this file is the environment-dependency of the transformation code, this whole
folder need to put into this path: /home/llvm/llvm/tools/clang/tools, and in this fold, you will find another CMakeLists.txt, you need to append one line to the end of the file. add_clang_subdirectory(sm-centric) 

And you can see there is a folder named Testcase. it includes 3 sub-folders, named MAM,MM and Vector. These 3 folds contain test cases. MM and MAM is the test case provided by professor and the Vector is the test case I designed.
In Vector folder. you can see there are two cuda files named with vector_add_org.cu and vector_add_org_smc.cu.
From the name, you can see the vector_add_org.cu is the original version of cuda code which means it's the code who need to be transformed.
and the vector_add_org_smc.cu is the code after transformation. 

================================================================
The instruction of how to compile it

To compile the project, you just need to follow the first part's hint, put these codes into the right path, and then 	Direct to the build-release folder(/home/ubuntu/llvm/build-release)
and run the ninja sm-centric

================================================================
The instruction of how to install it

After compile these code, you need to choose which test case you wanna use, and get the test case's path,  and direct to the build-release folder(/home/ubuntu/llvm/build-release)
run bin/sm-centric test_case's path
Then the after-transform CUDA code will be generated in the same folder with the original code.

================================================================
The understanding of output

In the output file (*****_smc.cu), it's the code after the transformation from CUDA to SM-Centric format. In the above of the output file, there will generate a new statement: #include "smc.h"
And at the beginning of the definition of the kernel function, there will be added " __SMC_Begin" shows the kernel function's beginning, and at the end of th definition of the kernel function,
there will be added " __SMC_End".

for the augument list of the definition of the kernel function, there will be added new arguments to the end of it.
it should be added: " dim3 __SMC_orgGridDim, int __SMC_workersNeeded, int *__SMC_workerCount, int * __SMC_newChunkSeq, int * __SMC_seqEnds"

And if there is a references of blockIdx.x, it will be replaced with the following "(int)fmodf((float)__SMC_chunkID, (float)__SMC_orgGridDim.x);"

The references of blockIdx.y will be replaced with "(int)(__SMC_chunkID/__SMC_orgGridDim.x);"

And the call of function frid(...) should be replaced with "dim3 __SMC_orgGridDim(...)" and others part should not be changed.

When get the call of the GPU kernel function, it will be added the " __SMC_init();" to the right before the call.

Finally for the GPU kernel function call, it will be appebded the following arguments to the ends of the call: "__SMC_orgGridDim, __SMC_workersNeeded, __SMC_workerCount, __SMC_newChunkSeq, __SMC_seqEnds"

