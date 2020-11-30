
switch (fun_id)
{
	case 0: // -(x1 x2 + y1 y2 + z1 z2)
		res = _mm512_fnmadd_pd(_x1, _x2, zeros);
		res = _mm512_fnmadd_pd(_y1, _y2, res);
		res = _mm512_fnmadd_pd(_z1, _z2, res);
		break;
	case 10: // -(x1 x2 + y1 y2)
		res = _mm512_fnmadd_pd(_x1, _x2, zeros);
		res = _mm512_fnmadd_pd(_y1, _y2, res);
		break;
	case 1: // -log[1 - (x1 x2 + y1 y2 + z1 z2)]  or  +inf 
    {
		res = _mm512_set1_pd(1.);
		res = _mm512_fnmadd_pd(_x1, _x2, res);
		res = _mm512_fnmadd_pd(_y1, _y2, res);
		res = _mm512_fnmadd_pd(_z1, _z2, res);

		const __mmask8 msk = _mm512_cmple_pd_mask(zeros, res); // res >= 0
		res = _mm512_mask_log_pd(_minf, msk, res);
		res = _mm512_sub_pd(zeros, res);
		break;
    }
	case 101: // 1 - (x1 x2 + y1 y2 + z1 z2)
		res = _mm512_set1_pd(1.);
		res = _mm512_fnmadd_pd(_x1, _x2, res);
		res = _mm512_fnmadd_pd(_y1, _y2, res);
		res = _mm512_fnmadd_pd(_z1, _z2, res);
		break;
	case 2: // sqrt[1 + (x1-x2)^2 + (y1-y2)^2]
    {
		res = _mm512_set1_pd(1.);

		const __m512d dx = _mm512_sub_pd(_x1, _x2);
		res = _mm512_fmadd_pd(dx, dx, res);

		const __m512d dy = _mm512_sub_pd(_y1, _y2);
		res = _mm512_fmadd_pd(dy, dy, res);
		res = _mm512_sqrt_pd(res);
		break;
    }
	case 3: // -sqrt[1 - (x1-x2)^2 - (y1-y2)^2]
    {
		res = _mm512_set1_pd(1.);
		const __m512d dx = _mm512_sub_pd(_x1, _x2);
		res = _mm512_fnmadd_pd(dx, dx, res);

		const __m512d dy = _mm512_sub_pd(_y1, _y2);
		res = _mm512_fnmadd_pd(dy, dy, res);

    	const __mmask8 msk = _mm512_cmple_pd_mask(zeros, res); // res >= 0
		res = _mm512_mask_sqrt_pd(_minf, msk, res);
		res = _mm512_sub_pd(zeros, res);
		break;
    }
	case 4: // sqrt[(x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2]
    {
		const __m512d dx = _mm512_sub_pd(_x1, _x2);
		res = _mm512_mul_pd(dx, dx);

		const __m512d dy = _mm512_sub_pd(_y1, _y2);
		res = _mm512_fmadd_pd(dy, dy, res);

		const __m512d dz = _mm512_sub_pd(_z1, _z2);
		res = _mm512_fmadd_pd(dz, dz, res);
		res = _mm512_sqrt_pd(res);
		break;
    }
	case 5: // -sqrt[(z1-z2)^2 - (x1-x2)^2 - (y1-y2)^2]
    {
		const __m512d dz = _mm512_sub_pd(_z1, _z2);
		res = _mm512_mul_pd(dz, dz);

		const __m512d dx = _mm512_sub_pd(_x1, _x2);
		res = _mm512_fnmadd_pd(dx, dx, res);

		const __m512d dy = _mm512_sub_pd(_y1, _y2);
		res = _mm512_fnmadd_pd(dy, dy, res);
		
		const __mmask8 msk = _mm512_cmple_pd_mask(zeros, res); // res >= 0
		res = _mm512_mask_sqrt_pd(_minf, msk, res);
		res = _mm512_sub_pd(zeros, res);
		break;
    }
}