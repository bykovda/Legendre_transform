/*
TODO: rename to  fenchel

lap_sqm_mex(x1, y1, z1, x2, y2, z2, g2, fun_id, sign, ) calculates
R_j = max_j {g2[j]-f(xyz1[i],xyz2[j])}    when sign = +1
R_j = min_j {g2[j]-f(xyz1[i],xyz2[j])}    when sign = -1

returns ind = argmax(argmin)

where f is defined by fun_id:
fun_id = 0  :	f = -(x1 x2 + y1 y2 + z1 z2)
fun_id = 10 :  	f = -(x1 x2 + y1 y2)
fun_id = 1  :  	f = -log[1 - (x1 x2 + y1 y2 + z1 z2)]
fun_id = 101:  	f =  1 - (x1 x2 + y1 y2 + z1 z2)   (see the Note below)
fun_id = 2	:	f =  sqrt(1 + (x1-x2)^2+(y1-y2)^2)
fun_id = 3	:	f =  -sqrt(1 - [(x1-x2)^2+(y1-y2)^2])
fun_id = 4	:	f =  sqrt((x1-x2)^2+(y1-y2)^2+(z1-z2)^2)

Note: when fun_id = 101 lap_sqm_mex calculates
max_j {g2[j]/f(xyz1[i],xyz2[j])}    when sign = +1
min_j {g2[j]/f(xyz1[i],xyz2[j])}    when sign = -1

*/

#include <math.h>
#include <matrix.h>
#include <mex.h>
#include <float.h>

#include <numeric>
#include <limits>
#include <utility>

#include <algorithm>

#include <omp.h>

#define simd_block 8
/* time for different simd_block size
1	1.4309
4   1.1398	1.1304
7   1.4112
8	1.1705	1.1637	1.1613
9 	1.2545	1.3008
14	1.4413
15	1.4735
16	1.2032	1.2046
32	1.3197
*/

#define fp double
#define inf std::numeric_limits<fp>::infinity()

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	// input data
	double* XS1 = mxGetPr(prhs[0]);
	double* YS1 = mxGetPr(prhs[1]);
	double* ZS1 = mxGetPr(prhs[2]);
	const int n1 = (int)mxGetM(prhs[0]);
	
	double* XS2 = mxGetPr(prhs[3]);
	double* YS2 = mxGetPr(prhs[4]);
	double* ZS2 = mxGetPr(prhs[5]);
	const int n2 = (int)mxGetM(prhs[3]);
	
	double* GS2 = mxGetPr(prhs[6]);
	
	const int fun_id = (int) *mxGetPr(prhs[7]);
	const fp sign = (fp) *mxGetPr(prhs[8]);
	
	// output data
	plhs[0] = mxCreateDoubleMatrix(n1, 1, mxREAL);
	double *R1 = (double*) mxGetData(plhs[0]);
	
	plhs[1] = mxCreateDoubleMatrix(n1, 1, mxREAL);//mxCreateNumericMatrix(n1, 1, mxUINT32_CLASS, mxREAL);
	double* ind = (double*)mxGetData(plhs[1]);
	//plhs[1] = mxCreateNumericMatrix(n1, 1, mxUINT32_CLASS, mxREAL);
	//unsigned int *ind = (unsigned int *) mxGetData(plhs[1]);
	
	const bool cf3D = !((fun_id == 10) || (fun_id == 2) || (fun_id == 3)); // 3D cost function
	
	// Copying xyz1 to make sure they are aligned & casting type to fp
	fp *xs1 = new fp[n1];
	fp *ys1 = new fp[n1];
	fp *zs1 = new fp[n1];
	for (int i = 0; i < n1; i++)
	{
		xs1[i] = (fp)XS1[i];
		ys1[i] = (fp)YS1[i];
		if (cf3D)
			zs1[i] = (fp)ZS1[i];
	}
	
	// Copying xyzg2 to bigger arrays with length being multiple of simd_block & casting type to fp
	const int n2_ext = ((n2 / simd_block) + 1) * simd_block;

	fp *xs2 = (fp*)_mm_malloc(n2_ext * sizeof(fp), 512); // = new fp[n2_ext];
	fp *ys2 = (fp*)_mm_malloc(n2_ext * sizeof(fp), 512);
	fp *zs2 = (fp*)_mm_malloc(n2_ext * sizeof(fp), 512);
	fp *gs2 = (fp*)_mm_malloc(n2_ext * sizeof(fp), 512);
	
	for (int i = 0; i < n2; i++)
	{
		xs2[i] = (fp)XS2[i];
		ys2[i] = (fp)YS2[i];
		if (cf3D)
			zs2[i] = (fp)ZS2[i];
		gs2[i]  = (fp)GS2[i];
	}
	
	#pragma omp parallel for
	for (int i = 0; i < n1; i++)
	{
		fp x1 = xs1[i];
		fp y1 = ys1[i];
		fp z1 = 0;
		if (cf3D)
			z1 = zs1[i];
		
//		fp* As = new fp[simd_block];
		fp* As = (fp*)_mm_malloc(simd_block * sizeof(fp), 512);
		bool* Ainf = (bool*)_mm_malloc(simd_block * sizeof(bool), 512);
		
		fp best_val;
		if (sign < 0)
			best_val = +inf;
		else
			best_val = -inf;
		int best_ind = -1;
		
        fp a;
		for (int j = 0; j < n2; j++)
		{
			if (j % simd_block == 0)
			{
				#include "auction_cost2.cpp"
				if (fun_id == 101)
				{
					#pragma omp simd
					for (int dj = 0; dj < simd_block; dj++)
						As[dj] = gs2[j + dj] / As[dj]; //???
				}
				else
				{
					#pragma omp simd
					for (int dj = 0; dj < simd_block; dj++)
						As[dj] = gs2[j + dj] - As[dj]; //???
				}
			}
			if (Ainf[j % simd_block])
			{
				if (sign < 0)
				{
					best_val = -inf;
					best_ind = j;
				}
			}
			else
			{
				a = As[j % simd_block];
				if ( ((sign < 0) && (a < best_val)) || ((sign > 0) && (a > best_val)) )
				{
					best_val = a;
					best_ind = j;
				}
			}
		}
		R1[i] = best_val;
		ind[i] = best_ind + 1; // Converting to MATLAB format
		_mm_free(As);
        _mm_free(Ainf);
	}
	delete[] xs1;
	delete[] ys1;
	delete[] zs1;
	_mm_free(xs2);
	_mm_free(ys2);
	_mm_free(zs2);
	_mm_free(gs2);
}
