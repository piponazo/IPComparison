/*!
 *      @file  invert_cuda.h
 *    @author  Luis Diaz Mas (LDM), piponazo@plagatux.es
 *
 *  @internal
 *    Created  13/08/12
 *   Revision 08/14/12 - 00:01:22
 *   Compiler  gcc/g++
 *        Web  http://plagatux.es
 *  Copyright  Copyright (c) 2012, Luis Diaz Mas
 *
 * This source code is released for free distribution under the terms of the
 * GNU General Public License as published by the Free Software Foundation.
 */

#ifndef  INVERT_CUDA_INC
#define  INVERT_CUDA_INC
void invert(const int w, const int h, const int c, unsigned char *devMem);
#endif   // ----- #ifndef INVERT_CUDA_INC  -----
