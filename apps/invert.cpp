/*!
 *      @file  invert.cpp
 *     @brief  Invert image
 *    @author  Luis Diaz Mas (LDM), piponazo@plagatux.es
 *
 *  @internal
 *    Created  08/08/12
 *   Revision 08/08/12 - 18:39
 *   Compiler  gcc/g++
 *    Company  
 *  Copyright  Copyright (c) 2012, Luis Diaz Mas
 *
 * This source code is released for free distribution under the terms of the
 * GNU General Public License as published by the Free Software Foundation.
 */

#include <argtable2.h>
#include <iostream>
#include <cstdlib>
#include <emmintrin.h>

#include <opencv2/core/core.hpp> // It has a typedef for uchar
#include <opencv2/highgui/highgui.hpp>

using namespace std;

struct arg_str *path = arg_str1("i", "image", "PATH",
  "Image path. It could be a gray-scale or color image");
struct arg_lit *show = arg_lit0("s", "show",
  "Show the original and the modified images");
struct arg_lit *h    = arg_lit0("h", "help",
  "Show this help");
struct arg_end *e    = arg_end(3);

void *argtable[] = {path , show, h, e};

void invertImageCPU(const uchar * __restrict__ src,
  uchar * __restrict__ dst, const int pixels);
void invertImageMMX(const uchar * __restrict__ src,
  uchar * __restrict__ dst, const int pixels);
void invertImageSSE(const uchar * __restrict__ src,
  uchar * __restrict__ dst, const int pixels);

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
    cerr << "There was " << nerrors << " errors in the command line parsed\n";
    arg_print_syntax(stdout, argtable, "\n");
    arg_print_glossary(stdout, argtable, "%-25s %s\n");
    exit(EXIT_FAILURE);
  }

  const int64 ticksSecond = cv::getTickFrequency();
  cv::Mat iImg, imgDstCPU, imgDstCV, imgDstMMX, imgDstSSE;
  iImg = cv::imread(path->sval[0], CV_LOAD_IMAGE_ANYCOLOR);
  imgDstCPU.create(iImg.rows, iImg.cols, iImg.type());
  imgDstCV.create(iImg.rows, iImg.cols, iImg.type());
  imgDstMMX.create(iImg.rows, iImg.cols, iImg.type());
  imgDstSSE.create(iImg.rows, iImg.cols, iImg.type());

  const int nPixels=iImg.cols*iImg.rows*iImg.channels();
  double t=0;
  cout << "Image size: " << iImg.cols << "x" << iImg.rows << "x"
       << iImg.channels()<< " = " << nPixels << endl;

  t = (double)cv::getTickCount();
  cv::bitwise_not(iImg, imgDstCV);
  t = ((double)cv::getTickCount() - t)/ticksSecond;
  cout << "OCV Avg/total: " << t << endl;

  t = (double)cv::getTickCount();
  invertImageCPU(iImg.data, imgDstCPU.data, nPixels);
  t = ((double)cv::getTickCount() - t)/ticksSecond;
  cout << "CPU Avg/total: " << t << endl;

  t = (double)cv::getTickCount();
  invertImageMMX(iImg.data, imgDstMMX.data, nPixels);
  t = ((double)cv::getTickCount() - t)/ticksSecond;
  cout << "MMX Avg/total: " << t << endl;

  t = (double)cv::getTickCount();
  invertImageSSE(iImg.data, imgDstSSE.data, nPixels);
  t = ((double)cv::getTickCount() - t)/ticksSecond;
  cout << "SSE Avg/total: " << t << endl;

  if (show->count==1)
  {
    cv::imshow("source", iImg);
    cv::imshow("CPU", imgDstCPU);
    cv::imshow("MMX", imgDstMMX);
    cv::imshow("SSE", imgDstSSE);
    cv::waitKey(0);
    cv::destroyAllWindows();
  }
  arg_freetable(argtable, sizeof(argtable)/sizeof(argtable[0]));
}

void invertImageCPU(const uchar * __restrict__ src,
  uchar * __restrict__ dst, const int pixels)
{
  register int i;
  for (i = 0; i < pixels; i++ )
    *dst++ = 255 - *src++;
}

void invertImageMMX(const uchar * __restrict__ src,
  uchar * __restrict__ dst, const int pixels)
{
  // 8 pixels are processed in one loop
  register int i;
  int nLoop = pixels/8;
  __m64* pIn = (__m64*) src;          // input pointer
  __m64* pOut = (__m64*) dst;         // output pointer
  __m64 tmp;                          // work variable
  _mm_empty();                        // Empties the multimedia state
  __m64 n1 = _mm_set_pi32(~0, ~0);
  for (i = 0; i < nLoop; ++i)
  {
    tmp = _mm_subs_pu8 (n1 , *pIn);   // Unsigned subtraction with saturation.
                                      // tmp = n1 - *pIn  for each byte.
    *pOut = tmp;
    pIn++;                            // next 8 pixels
    pOut++;
  }
  _mm_empty();                        // Empties the multimedia state
}

void invertImageSSE(const uchar * __restrict__ src,
  uchar * __restrict__ dst, const int pixels)
{
  // 16 pixels are processed in one loop
  register int i;
  int nLoop = pixels/16;
  __m128i* pIn = (__m128i*) src;      // input pointer
  __m128i* pOut = (__m128i*) dst;     // output pointer
  __m128i tmp;                        // work variable
  _mm_empty();                        // Empties the multimedia state
  __m128i n1 = _mm_set1_epi32(~0);
  for (i = 0; i < nLoop; ++i)
  {
    tmp = _mm_subs_epi8 (n1 , *pIn);  // Unsigned subtraction with saturation.
                                      // tmp = n1 - *pIn  for each byte
    *pOut = tmp;
    pIn++;                            // next 16 pixels
    pOut++;
  }
  _mm_empty();                        // Empties the multimedia state
}
