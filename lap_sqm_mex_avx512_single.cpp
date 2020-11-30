/*
TODO: rename to fenchel

lap_sqm_mex(x1, y1, z1, x2, y2, z2, g2, fun_id, sign) calculates
R[j] = max_j {g2[j] - f(xyz1[i], xyz2[j])}	when sign = +1
R[j] = min_j {g2[j] - f(xyz1[i], xyz2[j])}	when sign = -1

returns R and ind = argmax/argmin

The kernel f is defined by fun_id:
fun_id = 0  :	f = -(x1 x2 + y1 y2 + z1 z2)
fun_id = 10 :  	f = -(x1 x2 + y1 y2)
fun_id = 1  :  	f = -log[1 - (x1 x2 + y1 y2 + z1 z2)]	(and +infinity if log-argument is negative)
fun_id = 101:  	f =  1 - (x1 x2 + y1 y2 + z1 z2)		(see the Note below)
fun_id = 2	:	f =  sqrt[1 + (x1-x2)^2 + (y1-y2)^2]
fun_id = 3	:	f = -sqrt[1 - (x1-x2)^2 - (y1-y2)^2]	(and +infinity if sqrt-argument is negative)
fun_id = 4	:	f =  sqrt[(x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2]
fun_id = 5	:	f = -sqrt[(z1-z2)^2 - (x1-x2)^2 - (y1-y2)^2]	(and +infinity if sqrt-argument is negative)

Note: when fun_id = 101 the function calculates
max_j {g2[j] / f(xyz1[i], xyz2[j])}	when sign = +1
min_j {g2[j] / f(xyz1[i], xyz2[j])}	when sign = -1

*/
#include <mex.h>
#include <limits>
#include <omp.h>
#include <immintrin.h>

#define simd_block 16
#define fp float
#define inf std::numeric_limits<fp>::infinity()

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
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

	const int fun_id = (int)*mxGetPr(prhs[7]);
	const fp sign = (fp)*mxGetPr(prhs[8]);
	const bool to_max = sign > 0;

	// output data
	plhs[0] = mxCreateDoubleMatrix(n1, 1, mxREAL);
	double* R1 = (double*)mxGetData(plhs[0]);

	plhs[1] = mxCreateDoubleMatrix(n1, 1, mxREAL);
	double* ind = (double*)mxGetData(plhs[1]);
	//plhs[1] = mxCreateNumericMatrix(n1, 1, mxUINT32_CLASS, mxREAL);
	//unsigned int *ind = (unsigned int *) mxGetData(plhs[1]);

const bool cf3D = !((fun_id == 10) || (fun_id == 2) || (fun_id == 3)); // 3D cost function

	// Copying xyz1 to make sure they are aligned & casting type to fp
	fp* xs1 = (fp*)_mm_malloc(n1 * sizeof(fp), 64);
	fp* ys1 = (fp*)_mm_malloc(n1 * sizeof(fp), 64);
	fp* zs1 = cf3D ? (fp*)_mm_malloc(n1 * sizeof(fp), 64) : NULL;

	if (!xs1 || !ys1 || (cf3D && !zs1))
		return;

	for (int i = 0; i < n1; i++)
	{
		xs1[i] = (fp)XS1[i];
		ys1[i] = (fp)YS1[i];
	}
	if (cf3D)
		for (int i = 0; i < n1; i++)
			zs1[i] = (fp)ZS1[i];

	// Copying xyzg2 to bigger arrays with length being a multiple of simd_block & casting type to fp
	const int n_blocks = 1 + (n2 - 1) / simd_block;
	const int n2_ext = n_blocks * simd_block;

	fp* xs2 = (fp*)_mm_malloc(n2_ext * sizeof(fp), 64); // = new fp[n2_ext];
	fp* ys2 = (fp*)_mm_malloc(n2_ext * sizeof(fp), 64);
	fp* zs2 = cf3D ? (fp*)_mm_malloc(n2_ext * sizeof(fp), 64) : NULL;
	fp* gs2 = (fp*)_mm_malloc(n2_ext * sizeof(fp), 64);

	if (!xs2 || !ys2 || (cf3D && !zs2) || !gs2)
		return;

	for (int i = 0; i < n2; i++)
	{
		xs2[i] = (fp)XS2[i];
		ys2[i] = (fp)YS2[i];
		gs2[i] = (fp)GS2[i];
	}
	for (int i = n2; i < n2_ext; i++)
	{
		xs2[i] = xs2[n2-1];
		ys2[i] = ys2[n2-1];
		gs2[i] = gs2[n2-1];
	}

	if (cf3D)
	{
		for (int i = 0; i < n2; i++)
			zs2[i] = (fp)ZS2[i];
		for (int i = n2; i < n2_ext; i++)
			zs2[i] = (fp)zs2[n2-1];
	}

	#pragma omp parallel for                
	for (int i = 0; i < n1; i++)
	{
		const __m512 _x1 = _mm512_set1_ps(xs1[i]);
		const __m512 _y1 = _mm512_set1_ps(ys1[i]);
		const __m512 _z1 = cf3D ? _mm512_set1_ps(zs1[i]) : __m512();

		__m512 _best_val = _mm512_set1_ps(to_max ? -inf : inf);
		__m512i _best_block = _mm512_set1_epi32(-1);

		const __m512 _minf = _mm512_set1_ps(-inf);
		const __m512 zeros = _mm512_setzero_ps();

		for (int jb = 0; jb < n_blocks; jb++)
		{
			int j = jb * simd_block;
			const __m512 _x2 = _mm512_load_ps(&xs2[j]);
			const __m512 _y2 = _mm512_load_ps(&ys2[j]);
			const __m512 _z2 = cf3D ? _mm512_load_ps(&zs2[j]) : __m512();
			__m512 res;

			#include "cost_fun_avx512_single.cpp"

			const __m512 _g = _mm512_load_ps(&gs2[j]);
			if (fun_id == 101)
				res = _mm512_div_ps(_g, res);
			else
				res = _mm512_sub_ps(_g, res);

			const __mmask16 msk2 = _mm512_cmplt_ps_mask(_best_val, res); // best_val < res 
			if (to_max)
			{
				_best_val = _mm512_mask_blend_ps(msk2, _best_val, res);
				_best_block = _mm512_mask_blend_epi32(msk2, _best_block, _mm512_set1_epi32(jb));
			}
			else
			{
				_best_val = _mm512_mask_blend_ps(msk2, res, _best_val);
				_best_block = _mm512_mask_blend_epi32(msk2, _mm512_set1_epi32(jb), _best_block);
			}
		}

		fp* As = (fp*)_mm_malloc(simd_block  * sizeof(fp), 64);
		long int* Is = (long int*)_mm_malloc(simd_block  * sizeof(long int), 64);
		_mm512_store_ps(As, _best_val);
		_mm512_store_epi32(Is, _best_block);

		fp best_val = to_max ? -inf : +inf;
		int best_ind = -1;
		for (int j = 0; j < simd_block; j++)
		{
			fp a = As[j];
			if (((!to_max) && (a < best_val)) || (to_max && (a > best_val)))
			{
				best_val = a;
				best_ind = Is[j] * simd_block + j;
			}
		}
		if (best_ind > n2 - 1)
			best_ind = n2 - 1;
		R1[i] = (double)best_val;
		ind[i] = best_ind + 1; // Converting to MATLAB format
		_mm_free(As);
		_mm_free(Is);
	}
	_mm_free(xs1);
	_mm_free(xs2);
	_mm_free(ys1);
	_mm_free(ys2);
	if (cf3D)
	{
		_mm_free(zs1);
		_mm_free(zs2);
	}
	_mm_free(gs2);
}
