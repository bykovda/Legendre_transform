memset(Ainf, 0, simd_block * sizeof(bool));

switch (fun_id)
{
	case 0: //3D quadratic
		#pragma omp simd
		for (int dj = 0; dj < simd_block; dj++)
			As[dj] = -(x1*xs2[j+dj] + y1*ys2[j+dj] + z1*zs2[j+dj]); //added minus
		break;
	case 10: //2D quadratic
		#pragma omp simd
		for (int dj = 0; dj < simd_block; dj++)
			As[dj] = -(x1*xs2[j+dj] + y1*ys2[j+dj]); //added minus
		break;
	case 1: //reflector
		#pragma omp simd
		for (int dj = 0; dj < simd_block; dj++)
			As[dj] = 1.0 - (x1*xs2[j+dj] + y1*ys2[j+dj] + z1*zs2[j+dj]);
		#pragma omp simd
		for (int dj = 0; dj < simd_block; dj++)
            if (As[dj] < 0)
            {
                Ainf[dj] = true;
                As[dj] = 1.0;
            }
		#pragma omp simd
		for (int dj = 0; dj < simd_block; dj++)
			As[dj] = -log(As[dj]); // added minus
		break;
	case 101: //reflector without log
		#pragma omp simd
		for (int dj = 0; dj < simd_block; dj++)
			As[dj] = 1.0 - (x1*xs2[j+dj] + y1*ys2[j+dj] + z1*zs2[j+dj]);
		#pragma omp simd
		for (int dj = 0; dj < simd_block; dj++)
			Ainf[dj] = As[dj] < 0;
		break;
	case 2: //collimated beam shaping with 2 plano-freeform lenses || free-form thin-element eikonal
		#pragma omp simd
		for (int dj = 0; dj < simd_block; dj++)
		{
			fp dx = x1-xs2[j+dj];
			fp dy = y1-ys2[j+dj];
			As[dj] = sqrt(1.0 + dx*dx + dy*dy);
		}
		break;
	case 3: //collimated beam shaping with 1 freeform-freeform lens
		#pragma omp simd
		for (int dj = 0; dj < simd_block; dj++)
		{
			fp dx = x1-xs2[j+dj];
			fp dy = y1-ys2[j+dj];
			As[dj] = 1.0 - (dx*dx + dy*dy);
		}
		#pragma omp simd
		for (int dj = 0; dj < simd_block; dj++)
            if (As[dj] < 0)
            {
                Ainf[dj] = true;
                As[dj] = 0.0;
            }
		#pragma omp simd
		for (int dj = 0; dj < simd_block; dj++)
			As[dj] = -sqrt(As[dj]); //added minus
		break;
	case 4: //eikonal on curvilinear surfaces
		#pragma omp simd
		for (int dj = 0; dj < simd_block; dj++)
		{
			fp dx = x1-xs2[j+dj];
			fp dy = y1-ys2[j+dj];
			fp dz = z1-zs2[j+dj];
			As[dj] = sqrt(dx*dx + dy*dy + dz*dz);
		}
		break;
}