/*!
 *      @file  invert.cu
 *     @brief  Kernel for invert CUDA application
 *    @author  Luis Diaz Mas (LDM), piponazo@plagatux.es
 *
 *  @internal
 *    Created  08/08/12
 *   Revision 08/14/12 - 00:01:15
 *   Compiler  gcc/g++
 *        Web  http://plagatux.es
 *  Copyright  Copyright (c) 2012, Luis Diaz Mas
 *
 * This source code is released for free distribution under the terms of the
 * GNU General Public License as published by the Free Software Foundation.
 */

#include <cstdio>
#include <opencv2/core/core.hpp>

__global__ void invertKernel(unsigned char *mem, int w, int h, int channels)
{
  // Calculate our pixel's location
  const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  const int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int offset = (x + y * blockDim.x * gridDim.x)*channels;
  
  for (int i=0; i<channels; i++)
    mem[offset + i]     = 255 - mem[offset + i];
}

void invert(const int w, const int h, const int c, unsigned char *devMem)
{
  dim3 threads(16, 16);
  dim3 blocks(w/16, h/16);

  double t = (double)cv::getTickCount();
  invertKernel<<<blocks, threads>>>(devMem, w, h, c);
  t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
  printf("Time: %f secs\n", t);
}

