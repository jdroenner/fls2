
int position(int x, int y, int x_size){
	return y*x_size + x;
}


/**
This kernel assumes that mask and patched rasters have an equal size and the patch array is smaller or equally sized
*/
__kernel void masked_patch(__global float* patched_array, __global float* patch_array, __global const char* mask_array, const int current_mask, const int patched_x_size, const int patch_x_size, const int mask_x_size){
    int out_x_size = get_global_size(0);
	int out_y_size = get_global_size(1);
	int out_x_gid = get_global_id(0);
	int out_y_gid = get_global_id(1);
	int patched_gid = position(out_x_gid,out_y_gid,patched_x_size);
	int patch_gid = position(out_x_gid,out_y_gid,patch_x_size);
	int mask_gid = position(out_x_gid,out_y_gid,mask_x_size);

    char mask_value = mask_array[mask_gid];
    float patched_array_value = patched_array[patched_gid];
    float patch_array_value = patch_array[patch_gid];

	if (mask_value == current_mask){
	    //printf("x: %d, y: %d, mask_array: %d, current_mask: %d, patched_array: %f, patch_array: %f \n", out_x_gid, out_y_gid, mask_value, current_mask, patched_array_value, patch_array_value);
	    patched_array[patched_gid] = patch_array_value;
	}
}