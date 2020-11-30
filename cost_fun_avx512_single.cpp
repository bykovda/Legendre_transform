
switch (fun_id)
{
	case 0: // -(x1 x2 + y1 y2 + z1 z2)
		res = _mm512_fnmadd_ps(_x1, _x2, zeros);
		res = _mm512_fnmadd_ps(_y1, _y2, res);
		res = _mm512_fnmadd_ps(_z1, _z2, res);
		break;
	case 10: // -(x1 x2 + y1 y2)
		res = _mm512_fnmadd_ps(_x1, _x2, zeros);
		res = _mm512_fnmadd_ps(_y1, _y2, res);
		break;
	case 1: // -log[1 - (x1 x2 + y1 y2 + z1 z2)]  or  +inf 
    {
		res = _mm512_set1_ps(1.);
		res = _mm512_fnmadd_ps(_x1, _x2, res);
		res = _mm512_fnmadd_ps(_y1, _y2, res);
		res = _mm512_fnmadd_ps(_z1, _z2, res);

		const __mmask16 msk = _mm512_cmple_ps_mask(zeros, res); // res >= 0
		res = _mm512_mask_log_ps(_minf, msk, res);
		res = _mm512_sub_ps(zeros, res);
		break;
    }
	case 101: // 1 - (x1 x2 + y1 y2 + z1 z2)
		res = _mm512_set1_ps(1.);
		res = _mm512_fnmadd_ps(_x1, _x2, res);
		res = _mm512_fnmadd_ps(_y1, _y2, res);
		res = _mm512_fnmadd_ps(_z1, _z2, res);
		break;
	case 2: // sqrt[1 + (x1-x2)^2 + (y1-y2)^2]
    {
		res = _mm512_set1_ps(1.);

		const __m512 dx = _mm512_sub_ps(_x1, _x2);
		res = _mm512_fmadd_ps(dx, dx, res);

		const __m512 dy = _mm512_sub_ps(_y1, _y2);
		res = _mm512_fmadd_ps(dy, dy, res);
		res = _mm512_sqrt_ps(res);
		break;
    }
	case 3: // -sqrt[1 - (x1-x2)^2 - (y1-y2)^2]
    {
		res = _mm512_set1_ps(1.);
		const __m512 dx = _mm512_sub_ps(_x1, _x2);
		res = _mm512_fnmadd_ps(dx, dx, res);

		const __m512 dy = _mm512_sub_ps(_y1, _y2);
		res = _mm512_fnmadd_ps(dy, dy, res);

    	const __mmask16 msk = _mm512_cmple_ps_mask(zeros, res); // res >= 0
		res = _mm512_mask_sqrt_ps(_minf, msk, res);
		res = _mm512_sub_ps(zeros, res);
		break;
    }
	case 4: // sqrt[(x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2]
    {
		const __m512 dx = _mm512_sub_ps(_x1, _x2);
		res = _mm512_mul_ps(dx, dx);

		const __m512 dy = _mm512_sub_ps(_y1, _y2);
		res = _mm512_fmadd_ps(dy, dy, res);

		const __m512 dz = _mm512_sub_ps(_z1, _z2);
		res = _mm512_fmadd_ps(dz, dz, res);
		res = _mm512_sqrt_ps(res);
		break;
    }
	case 5: // -sqrt[(z1-z2)^2 - (x1-x2)^2 - (y1-y2)^2]
    {
		const __m512 dz = _mm512_sub_ps(_z1, _z2);
		res = _mm512_mul_ps(dz, dz);

		const __m512 dx = _mm512_sub_ps(_x1, _x2);
		res = _mm512_fnmadd_ps(dx, dx, res);

		const __m512 dy = _mm512_sub_ps(_y1, _y2);
		res = _mm512_fnmadd_ps(dy, dy, res);
		
		const __mmask16 msk = _mm512_cmple_ps_mask(zeros, res); // res >= 0
		res = _mm512_mask_sqrt_ps(_minf, msk, res);
		res = _mm512_sub_ps(zeros, res);
		break;
    }
}