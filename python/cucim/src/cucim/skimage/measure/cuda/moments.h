template<typename T>
__device__ void inertia_tensor_2x2(const T* mu, T* result){
	T mu0, mxx, mxy, myy;
	mu0 = mu[0];
	mxx = mu[6];
	myy = mu[2];
	mxy = mu[4];

	result[0] = myy / mu0;
	result[1] = result[2] = -mxy / mu0;
	result[3] = mxx / mu0;
}
