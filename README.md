# legendre_transform
Generalized Legendre (Legendre–Fenchel) transformation for applications in nonimaging optics

C implementation

function lap_sqm_mex(x1, y1, z1, x2, y2, z2, g2, fun_id, sign, ) calculates
R_j = max_j {g2[j]-f(xyz1[i],xyz2[j])}    when sign = +1
R_j = min_j {g2[j]-f(xyz1[i],xyz2[j])}    when sign = -1

returns ind = argmax(argmin)

where f is defined by fun_id:
fun_id = 0  :	f = -(x1 x2 + y1 y2 + z1 z2) $a^2+b^2=c^2$
fun_id = 10 :  	f = -(x1 x2 + y1 y2)
fun_id = 1  :  	f = -log[1 - (x1 x2 + y1 y2 + z1 z2)]
fun_id = 101:  	f =  1 - (x1 x2 + y1 y2 + z1 z2)   (see the Note below)
fun_id = 2	:	f =  sqrt(1 + (x1-x2)^2+(y1-y2)^2)
fun_id = 3	:	f =  -sqrt(1 - [(x1-x2)^2+(y1-y2)^2])
fun_id = 4	:	f =  sqrt((x1-x2)^2+(y1-y2)^2+(z1-z2)^2)
Note: when fun_id = 101 lap_sqm_mex calculates
max_j {g2[j]/f(xyz1[i],xyz2[j])}    when sign = +1
min_j {g2[j]/f(xyz1[i],xyz2[j])}    when sign = -1

MATLAB implementation allows to use additional parameters in the cost function
