int position(int x, int y, int x_size){
	return y*x_size + x;
}

int count_no_data(float *values, int number_of_values, float no_data){
    int counter = 0;

    for( int i = 0; i < number_of_values; i++){
        if (values[i] <= no_data){
            counter++;
        }
    }
    return counter;
}

float mean(float* values, int number_of_values, float no_data){
    int count = 0;
    float sum = 0;

    for( int i = 0; i < number_of_values; i++){
        if (values[i] <= no_data){
            count++;
            sum += values[i];
        }
    }
    return sum / count;
}

__kernel void bilinear(__global float *input, __global float *output, int input_x_size, int input_y_size, float no_data){
	// 0 1
	// 2 3
	float edges[4];

    int out_x_size = get_global_size(0);
	int out_y_size = get_global_size(1);
	int out_x_gid = get_global_id(0);
	int out_y_gid = get_global_id(1);
	int out_gid = position(out_x_gid,out_y_gid,out_x_size);


    float x_ratio = convert_float(input_x_size) / convert_float(out_x_size);
    float y_ratio = convert_float(input_y_size) / convert_float(out_y_size);
    //printf("x_ratio = %f, y_ratio = %f\n", x_ratio, y_ratio);
    int x_block_size = out_x_size / input_x_size;
    int y_block_size = out_y_size / input_y_size;


    if (out_x_gid < (x_block_size/2) || out_y_gid < (y_block_size/2) || out_x_gid > (out_x_size - x_block_size/2) || out_y_gid > (out_y_size - y_block_size/2)){
        output[out_gid] = no_data;
        return;
    }

    float fl_x_pos = out_x_gid * x_ratio - 0.5;
    float fl_y_pos = out_y_gid * y_ratio - 0.5;
    //printf("fl_x_pos = %f, fl_y_pos = %f\n", fl_x_pos, fl_y_pos);


    int in_x_gid = convert_int(fl_x_pos);
    int in_y_gid = convert_int(fl_y_pos);

    float rl_x_pos = fl_x_pos - in_x_gid;
    float rl_y_pos = fl_y_pos - in_y_gid;

    edges[0] = input[position(in_x_gid,   in_y_gid,   input_x_size)];
    //printf("%f = input[%d] (gid = %d, gid_x=%d, gid_y=%d)\n", aa, input[position(in_x_gid,   in_y_gid,   input_x_size)], position(in_x_gid,   in_y_gid,   input_x_size), in_x_gid, in_y_gid);
    edges[1] = input[position(in_x_gid+1, in_y_gid,   input_x_size)];
    edges[2] = input[position(in_x_gid,   in_y_gid+1, input_x_size)];
    edges[3] = input[position(in_x_gid+1, in_y_gid+1, input_x_size)];

    if (edges[0] <= no_data && edges[1] <= no_data && edges[3] <= no_data && edges[4] <= no_data){
        output[out_gid] = no_data;
        return;
    }

    float mean_val = mean(edges, 4, no_data);

    if (edges[0] <= no_data){
        edges[0] = mean_val;
    }
    if (edges[1] <= no_data){
        edges[1] = mean_val;
    }
    if (edges[2] <= no_data){
        edges[2] = mean_val;
    }
    if (edges[3] <= no_data){
        edges[3] = mean_val;
    }

    float val_x1 = edges[0] * (1-rl_x_pos) + edges[1] * rl_x_pos;
    float val_x2 = edges[2] * (1-rl_x_pos) + edges[3] * rl_x_pos ;
    float res = val_x1 * (1-rl_y_pos) + val_x2 * rl_y_pos;


    output[out_gid] = res;
}