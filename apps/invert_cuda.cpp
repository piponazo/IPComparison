/*!
 *      @file  invert_cuda.cpp
 *     @brief  Invert image using CUDA C Runtime.
 *
 *    @author  Luis Diaz Mas (LDM), piponazo@plagatux.es
 *
 *  @internal
 *    Created  13/08/12
 *   Revision 08/14/12 - 09:58:21
 *   Compiler  gcc/g++
 *        Web  http://plagatux.es
 *  Copyright  Copyright (c) 2012, Luis Diaz Mas
 *
 * This source code is released for free distribution under the terms of the
 * GNU General Public License as published by the Free Software Foundation.
 */

#include <cstdio>
#include <argtable2.h>
#include "invert_cuda.h"
#include "cutil.h"
#include "cuda_runtime.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

struct arg_str *path = arg_str1("i", "image", "PATH",
  "Image path. It could be a gray-scale or color image");
struct arg_lit *show = arg_lit0("s", "show",
  "Show the original and the modified images");
struct arg_lit *h    = arg_lit0("h", "help",
  "Show this help");
struct arg_end *e    = arg_end(3);

void *argtable[] = {path , show, h, e};

int main(int argc, char **argv)
{
  // Read arguments
  int nerrors = arg_parse(argc, argv, argtable);
  if (h->count)
  {
    arg_print_syntax(stdout, argtable, "\n");
    arg_print_glossary(stdout, argtable, "%-25s %s\n");
    exit(EXIT_SUCCESS);
  }
  else if (nerrors!=0)
  {
    printf("There was %d errors in the command line parsed\n", nerrors);
    arg_print_syntax(stdout, argtable, "\n");
    arg_print_glossary(stdout, argtable, "%-25s %s\n");
    exit(EXIT_FAILURE);
  }

  CUDA_SAFE_CALL(cudaSetDevice(0));
  
  // Read input image
  cv::Mat iImg;
  iImg = cv::imread(path->sval[0], CV_LOAD_IMAGE_ANYCOLOR);
  unsigned int channels = iImg.channels();
  unsigned char *devMem;

  // Allocate device memory and copy input data to it
  CUDA_SAFE_CALL(cudaMalloc(&devMem, iImg.total()*channels));
  CUDA_SAFE_CALL(cudaMemcpy(devMem, iImg.data, iImg.total()*channels,
    cudaMemcpyHostToDevice) );

  // Call the CUDA wrapper
  invert(iImg.cols, iImg.rows, channels, devMem);

  // Show image if desired
  if (show->count)
  {
    // We have to copy from device memory to host image
    cv::Mat oImg(iImg.rows, iImg.cols, iImg.type());
    CUDA_SAFE_CALL(cudaMemcpy(oImg.data, devMem, iImg.total()*channels,
      cudaMemcpyDeviceToHost) );

    cv::imshow("source", iImg);
    cv::imshow("cuda", oImg);
    cv::waitKey(0);
  }

  CUDA_SAFE_CALL(cudaFree(devMem));
  CUDA_SAFE_CALL(cudaDeviceReset());

  return EXIT_SUCCESS;
}

