/*!
 *      @file  invert.cu
 *     @brief  Invert image using CUDA NPP library
 *    @author  Luis Diaz Mas (LDM), piponazo@plagatux.es
 *
 *  @internal
 *    Created  08/08/12
 *   Revision 08/14/12 - 09:17:49
 *   Compiler  gcc/g++
 *  Copyright  Copyright (c) 2012, Luis Diaz Mas
 *
 * This source code is released for free distribution under the terms of the
 * GNU General Public License as published by the Free Software Foundation.
 */

#include <cstdio>
#include <argtable2.h>
#include "cutil.h"
#include "npp.h"

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
  /////////////////////////////////////////////////////////
  // Read arguments
  /////////////////////////////////////////////////////////
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

  cv::Mat iImg;
  iImg = cv::imread(path->sval[0], CV_LOAD_IMAGE_ANYCOLOR);

  int step;
  Npp8u *devMemI=0;
  
  if (iImg.channels()==1)
    devMemI = nppiMalloc_8u_C1(iImg.cols, iImg.rows, &step);
  else if (iImg.channels()==3)
    devMemI = nppiMalloc_8u_C3(iImg.cols, iImg.rows, &step);
  else
  {
    perror("Not controlled case for channels != 1 && channels != 3");
    return EXIT_FAILURE;
  }

  assert (devMemI != 0);
  CUDA_SAFE_CALL(cudaMemcpy2D(devMemI, step, iImg.data, iImg.step1(),
    iImg.cols*iImg.channels(), iImg.rows, cudaMemcpyHostToDevice));

  NppiSize size;
  size.width = iImg.cols;
  size.height= iImg.rows;
  
  double t = (double)cv::getTickCount();
  if (iImg.channels()==1)
    nppiNot_8u_C1IR(devMemI, step, size);
  else
    nppiNot_8u_C3IR(devMemI, step, size);
  t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
  printf("Time: %f secs\n", t);

  if (show->count==1)
  {
    cv::Mat oImg;
    if (iImg.channels()==1)
      oImg.create(iImg.rows, iImg.cols, CV_8UC1);
    else
      oImg.create(iImg.rows, iImg.cols, CV_8UC3);

    CUDA_SAFE_CALL(cudaMemcpy2D(oImg.data, oImg.step1(), devMemI, step,
      iImg.cols*iImg.channels(), iImg.rows, cudaMemcpyDeviceToHost));

    cv::imshow("source", iImg);
    cv::imshow("npp", oImg);
    cv::waitKey(0);
  }
  arg_freetable(argtable, sizeof(argtable)/sizeof(argtable[0]));
  nppiFree(devMemI);
  return EXIT_SUCCESS;
}

