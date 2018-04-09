int position(int x, int y, int x_size){
	return y*x_size + x;
}

float mean(float *arr, int n, float no_data){
	float mean = 0.0f;
	float count = 0.0f;

	for(int i=0; i<n; i++){
		if(arr[i]>no_data){
			mean += arr[i];
			count += 1.0f;
		}
	}

	mean = mean/count;
	return mean;
}

float std(float *arr, int n, float average, float no_data){
	float sum = 0.0f;
	float count = 0.0f;

	for(int i=0; i<n; i++){
		if(arr[i]>no_data){
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
__kernel void replace_by_std(__global const float *input, __global float* output, float no_data) {
	int xsize = get_global_size(1);
	int ysize = get_global_size(0);
	int xgid = get_global_id(1);
	int ygid = get_global_id(0);
	int pos = position(xgid,ygid,xsize);

	output[pos] = input[pos]; // start with the input value

	float values[25];
	float sum = 0;
	int count = 0;


	for(int i=0; i<5; i++){
  		for(int j=0; j<5; j++){
  			float currentValue = no_data;
  			if(!(i==3 && j==3) && xgid+i-2>0 && ygid+j-2>0 && xgid+i-2<xsize && ygid+j-2<ysize){

  				currentValue = input[position(xgid+i-2,ygid+j-2,xsize)];
  				if( currentValue > no_data){
  					sum += currentValue;
  					count += 1;
  				}

  			}
  			values[i*5+j] = currentValue;
  		}
  	}

  	float mean = sum / count;
  	float std_value = std(values,25, mean, no_data);
  	float diff = input[pos] - mean;

 	//printf("mean= %f, std= %f, diff= %f\n", mean, std_value, diff);


  	if (fabs(diff) > 2*std_value){
  	    //printf("replace: value= %f, mean= %f, std= %f, diff= %f\n", input[pos], mean, std_value, diff);

  	    output[pos] = mean;

  	}
}