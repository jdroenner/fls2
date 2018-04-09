uchar classify_local(const int a, const int b, const int c, const int eps){ //NOTE: NEVER compare unsigned and signed values!

  const int slope_a_b = a-b;
  const int slope_b_c = b-c;
  const int i_eps = 1*eps;
  //printf(" | slope_a_b: %d, slope_b_c: %d | ", slope_a_b, slope_b_c);

  //if (abs(slope_a_b) < eps && abs(slope_b_c) < eps) //NOTE: should be the default case
  //   return 5; //flat
  if (slope_a_b < -i_eps && slope_b_c >  i_eps)
     return 1; //min
  if (slope_a_b >  i_eps && slope_b_c < -i_eps)
     return 9; // max

  if (slope_a_b > i_eps || slope_b_c > i_eps){ // up
     if (slope_a_b < slope_b_c)
         return 2;// # left
     if (slope_a_b == slope_b_c)
         return 3;// # wp
     if (slope_a_b > slope_b_c)
         return 4;// # right
  }

  if (slope_a_b < -i_eps || slope_b_c < -i_eps){ // down
     if (slope_a_b < slope_b_c)
         return 6;// # left
     if (slope_a_b == slope_b_c)
         return 7;// # wp
     if (slope_a_b > slope_b_c)
         return 8;// # right
  }

  return 5; // flat
}


kernel
void histogram_classify_local(global int *block_histograms, const int eps, global uchar *classfied_histograms){
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int x_size = get_global_size(0);
  const int gid = y * x_size + x;
  //printf("y: %d, x: %d, gid: %d ", y, x, gid);

  int b = block_histograms[gid];
  int a = (x > 0) ? block_histograms[gid-1] : b;
  int c = (x < x_size-1) ? block_histograms[gid+1] : b;
  uchar class = classify_local(a,b,c,eps);
  //printf("a: %d, b: %d, c: %d, eps: %d, class: %d \n", a, b, c, eps, class);

  classfied_histograms[gid] = class;
}


kernel
void window_aggregate_histogram_block(global int *block_histograms, const int window_y, const int window_z, global int *aggregate_histograms){
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);
  const int x_size = get_global_size(0);
  const int y_size = get_global_size(1);
  const int z_size = get_global_size(2);
  const int gid = z * y_size * x_size + y * x_size + x;

  //printf("z: %d, y: %d, x: %d, gid: %d, window_y: %d, window_z: %d \n", z, y, x, gid, window_y, window_z);
  const int y_u = window_y;
  const int z_u = window_z;

  int sum = 0;
  for ( int offset_z = -z_u; offset_z <= z_u; offset_z++ ){ // NOTE: always be careful when comparing unsigned and signed types!
    for ( int offset_y = -y_u; offset_y <= y_u; offset_y++ ){
      int y_s = y + offset_y;
      int z_s = z + offset_z;
      //printf("    z_s: %d, y_s: %d, x: %d \n", z_s, y_s, x);


      if (0 <= y_s && 0 <= z_s && y_s < y_size && z_s < z_size ){
        int sid = z_s * y_size * x_size + y_s * x_size + x;
        sum += block_histograms[sid];
        //printf("    sid: %d, sum: %d \n", sid, sum);
      }

    }
  }

  aggregate_histograms[gid] = sum;
}


kernel
void merge_histogram_block(global int *block_histograms, const int merge_x, const int merge_y, global int *merge_histograms){
  const int hc = get_global_id(0);
  const int x = get_global_id(1);
  const int y = get_global_id(2);
  const int hc_size = get_global_size(0);
  const int x_size = get_global_size(1);
  const int y_size = get_global_size(2);
  const int gid = y * x_size * hc_size + x * hc_size + hc;

  //printf("z: %d, y: %d, x: %d, gid: %d \n", z, y, x, gid);
  const int x_size_s = x_size * merge_x;
  const int y_size_s = y_size * merge_y;
  //printf("hc: %d, x: %d, y: %d, hc_size: %d, x_size: %d, y_size: %d, x_size_s: %d, y_size_s: %d, gid: %d \n", hc, x, y, hc_size, x_size, y_size, x_size_s, y_size_s, gid);


  int sum = 0;
  for (int offset_y = 0; offset_y < merge_y; offset_y++){
    for (int offset_x = 0; offset_x < merge_x; offset_x++){
      int x_s = x * merge_x + offset_x;
      int y_s = y * merge_y + offset_y;
      int sid = y_s * x_size_s * hc_size + x_s * hc_size + hc;

      sum += block_histograms[sid];

      //printf(" gid: %d,  x: %d, y: %d, x_s: %d, y_s: %d, sid: %d, sum: %d \n", gid, x, y, x_s, y_s, sid, sum);
    }
  }

  merge_histograms[gid] = sum;
}


kernel
void block_sum(global int *block_histograms, const int hist_size, global int *block_sum){
  const int c = get_global_id(0);
  const int x = get_global_id(1);
  const int y = get_global_id(2);

  const int c_size = get_global_size(0); // c = 4
  const int x_size = get_global_size(1); // x = 16
  const int y_size = get_global_size(2); // y = 11

  int gid = y * x_size * c_size + x * c_size + c;//w + x * w_size + y * x_size * w_size ;

  //printf("c: %d, x: %d, y: %d, gid: %d \n", c, x, y, gid);

  int sum = 0;
  for (int i = 0; i < hist_size; i++){

    int sid = gid * hist_size + i;
    sum += block_histograms[sid];

    //printf("i: %d, c: %d, x: %d, y: %d, gid: %d, sid: %d, sum: %d \n", i, c, x, y, gid, sid, sum);

  }

  //printf("c: %d, x: %d, y: %d, gid: %d, sum: %d \n", c, x, y, gid, sum);


  block_sum[gid] = sum;
}



kernel
void histogram_block_with_masks(global float *img, global uchar *img_claas, const int img_width, const int img_height, const int num_of_class, const float hist_min_value, const int hist_num_of_bins, const float hist_bin_width,  global int *histogram, local int *tmp_histogram)
{

//const int num_of_class = 4;
//const float hist_min_value = -25;
//const int hist_num_of_bins = 120;
//const float hist_bin_width = 0.25;


  const int histograms_buf_size = hist_num_of_bins * num_of_class;
  //local int tmp_histogram[4*100];// TODO: use a local kernel argument!
  //printf("img_width: %d, img_height: %d, histograms_buf_size: %d \n", img_width, img_height, histograms_buf_size);

  const int local_size = (int)get_local_size(0) * (int)get_local_size(1); //work-group size
  const int group_indx = get_group_id(1) * get_num_groups(0) + get_group_id(0);// * histograms_buf_size; //????
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int x_size = get_global_size(0);
  const int gid = y * x_size + x;
  const int tid = get_local_id(1) * get_local_size(0) + get_local_id(0);
  //printf("local_size: %d, group_indx: %d, x: %d, y: %d, gid: %d, tid: %d \n", local_size, group_indx, x, y, gid, tid);


  int j = histograms_buf_size; // histogram size * masks = area to initialize with zero -> upper limit (needed to fill)
  int indx = 0; // lower limit -> already filled
  // clear the local buffer that will generate the partial histogram
  do{
      if (tid < j)
          tmp_histogram[indx+tid] = 0;
      j -= local_size;
      indx += local_size;
  } while (j > 0);

  // sync all threads of the same group
  barrier(CLK_LOCAL_MEM_FENCE);

  if ((x < img_width) && (y < img_height))
  {
      const int px_id = img_width * y + x; // get the pix id!
      const float value = img[px_id];
      const uchar mask_class = img_claas[px_id];
      //printf("a gid: %d, value: %f, mask_class: %d \n", gid, value, mask_class);

      if (mask_class > 0) { // hardcoded class 0 = nodata / ignore
        const int hist_bin_index =  floor((value - hist_min_value) / hist_bin_width);
        if ((hist_bin_index >= 0) && (hist_bin_index < hist_num_of_bins)){
          const int buf_bin_index = hist_bin_index + (mask_class-1) * hist_num_of_bins;
          //printf("b gid: %d, value: %f, mask_class: %d, hist_bin_index: %d, buf_bin_index: %d \n", gid, value, mask_class, hist_bin_index, buf_bin_index);
          if (mask_class > num_of_class) {
            //printf("c x: %d, y: %d, gid: %d, value: %f, class: %d, hist_bin_index: %d, buf_bin_index: %d \n", x, y, gid, value, mask_class, hist_bin_index, buf_bin_index);
          }

          atomic_inc(&tmp_histogram[buf_bin_index]);
        }
        /*
        else {
          printf("OUT OF BOUNDS gid: %d, value: %f, mask_class: %d, hist_bin_index: %d \n", gid, value, mask_class, hist_bin_index);

        }
        */
      }

      else {
        //printf("0 CLASS gid: %d, value: %f, mask_class: %d\n", gid, value, mask_class);

      }

  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // copy the partial histogram to appropriate location in histogram given by group_indx
  if (local_size >= (histograms_buf_size)){
      if (tid < (histograms_buf_size))
          histogram[group_indx * histograms_buf_size + tid] = tmp_histogram[tid];
  }
  else{
      j = histograms_buf_size;
      indx = 0;
      do
      {
          if (tid < j)
              histogram[group_indx * histograms_buf_size + indx + tid] = tmp_histogram[indx + tid];
            j -= local_size;
          indx += local_size;
      } while (j > 0);
  }
}
