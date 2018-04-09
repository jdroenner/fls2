int position(int x, int y, int x_size){
	return y*x_size + x;
}

// -999.9f is missing value
float mean(float *arr, int n){
	float mean = 0.0f;
	float count = 0.0f;
	
	for(int i=0; i<n; i++){
		if(arr[i]>-999.0f){
			mean += arr[i];
			count += 1.0f;
		}
	}
	
	mean = mean/count;
	return mean;
}

// -999.9f is missing value
float std(float *arr, int n){
	float average = mean(arr, n);
	float sum = 0.0f;
	float count = 0.0f;
	
	for(int i=0; i<n; i++){
		if(arr[i]>-999.0f){
			float differenz = arr[i]-average;
			float power = pow(differenz,2.0f);
			sum += power;
			count += 1.0f;
		}
	}
	
	if(count<1){
		return 0.0f;
	}
	sum = sum/count;
	sum = sqrt(sum);
	return sum;
}


// Calculate standard deviation of 5x5 surrounding
__kernel void calcSTD(__global const float *input, __global float *res_g) {
	int xsize = get_global_size(1);
	int ysize = get_global_size(0);
	int xgid = get_global_id(1);
	int ygid = get_global_id(0);
	int pos = position(xgid,ygid,xsize);

	float values[25];

	for(int i=0; i<5; i++){
  		for(int j=0; j<5; j++){
  			float currentValue = -999.9f;
  			if(xgid+i-2>0 && ygid+j-2>0 && xgid+i-2<xsize && ygid+j-2<ysize){
  				currentValue = input[position(xgid+i-2,ygid+j-2,xsize)];
  			}
  			values[i*5+j] = currentValue;
  		}
  	}
  	
  	res_g[pos] = std(values,25);
}

// Calculate standard deviation of direct neighbours without center pixel value
__kernel void calcSTD_directNeighboursWithoutCenter(__global const float *input, __global float *res_g) {
	int xsize = get_global_size(1);
	int ysize = get_global_size(0);
	int xgid = get_global_id(1);
	int ygid = get_global_id(0);
	int pos = position(xgid,ygid,xsize);

	float values[9];

	for(int i=0; i<3; i++){
  		for(int j=0; j<3; j++){
  			float currentValue = -999.9f;
  			if(!(i==1 && j==1) && xgid+i-1>0 && ygid+j-1>0 && xgid+i-1<xsize && ygid+j-1<ysize){
  				currentValue = input[position(xgid+i-1,ygid+j-1,xsize)];
  			}
  			values[i*3+j] = currentValue;
  		}
  	}
  	
  	res_g[pos] = std(values,9);
}

// Filter 1-pixel-sized clouds
__kernel void filterSmallClouds(__global const float *input, __global float *res_g) {
	int xsize = get_global_size(1);
	int ysize = get_global_size(0);
	int xgid = get_global_id(1);
	int ygid = get_global_id(0);
	int pos = position(xgid,ygid,xsize);
	
	int neighbours = 0;
	
	if(input[pos]>0.0f){
		for(int i=0; i<3; i++){
  			for(int j=0; j<3; j++){
  				if(!(i==1 && j==1) && xgid+i-1>0 && ygid+j-1>0 && xgid+i-1<xsize && ygid+j-1<ysize && input[position(xgid+i-1,ygid+j-1,xsize)]>0.0f){
  					neighbours += 1;
  				}
  			}
  		}
	}
	if(neighbours>3){
		res_g[pos] = 1.0f;
	} else {
		res_g[pos] = 0.0f;
	}
}