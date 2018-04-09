int position2d(int x, int y, int x_size){
	return y*x_size + x;
}

int position3d(int x, int y, int z, int x_size, int y_size){
	return z * y_size * x_size + y*x_size + x;
}

float hist_bin_to_value(const int hist_bin, const float2 hist_range, const float hist_bin_width){
    const float hist_range_start, hist_range_end = hist_range;
    return hist_range_start+(hist_bin * hist_bin_width);
}

int hist_value_to_bin(const float value, const float2 hist_range, const float hist_bin_width){
    const float hist_range_start, hist_range_end = hist_range;
    return (int) ((value - hist_range_start)/hist_bin_width);
}


float slope(float x1, float x2, float v1, float v2){
    if (x1 == x2) {
        return 0.0f;
    } else {
        return 1.0f * (v2 - v1) / (x2 - x1);
    }
}


/*
this should be able to run in parallel. sync after read?
*/
int smooth_array(int prev_prev_class, int prev_class, int cur_class, int next_class, int next_next_class){
    const int smoothened = (prev_prev_class+prev_class*3+cur_class*6+next_class*3+next_next_class[i+2])/14;
    return smoothened;
}

kernel smooth_array_kernel(global int *in_class, global int *out_smooth_class, int* skip) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);
    const int x_size = get_global_size(0);
    const int y_size = get_global_size(1);
    const int z_size = get_global_size(2);

    const int cur_pos = position3d(x,y,z,x_size,y_size);
    const int skip_pos = position2d(y, z, y_size);
    int smooth_class;

    if (z < 2 || z > z_size-2 || skip[skip_pos] == 1){
        smooth_class = in_class[cur_pos];
    } else {
        smooth_class = smooth_array(in_class[cur_pos-2], in_class[cur_pos-1], in_class[cur_pos], in_class[cur_pos+1], in_class[cur_pos+2]);
    }
    out_smooth_class[cur_pos] = smooth_class;
}

/*
this can run in parallel for each bin from prev to next ignoring first and last bin!
*/
bool ripple_away(int prev_class, int cur_class, int next_class){
    if (cur_class == 9 && (prev_class == 1 || 5 <= prev_class && prev_class <= 8) && 1 <= next_class && next_class <= 5)
        return true;
    if (cur_class == 1 && (prev_class == 9 || 2 <= next_class && next_class <= 5) && 5 <= next_class && next_class <= 9)
        return true;
    return false;
}

kernel ripple_away_kernel(global int *in_class, global int *out_ripple_class, int* skip) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);
    const int x_size = get_global_size(0);
    const int y_size = get_global_size(1);
    const int z_size = get_global_size(2);

    const int cur_pos = position3d(x,y,z,x_size,y_size);
    const int skip_pos = position2d(y, z, y_size);

    if (z < 1 || z > z_size-1 || skip[skip_pos] == 1){
        return;
    }
    
    const bool ripple = ripple_away(in_class[cur_pos-1], in_class[cur_pos], in_class[cur_pos+1]);
    out_ripple_class[cur_pos] = (ripple)? 5 : in_class[cur_pos];
}

kernel ip_ripple_away_kernel(global int *in_out_class, global int* skip, const int hist_size) {
    const int y = get_global_id(0);
    const int z = get_global_id(1);
    //const int z = get_global_id(2);
    const int y_size = get_global_size(0);
    const int z_size = get_global_size(1);
    //const int z_size = get_global_size(2);

    const int skip_pos = position2d(y, z, y_size);

    if (z < 1 || z > z_size-1 || skip[skip_pos] == 1){
        return;
    }

    const int first_pos = position3d(0,y,z,hist_size,y_size);
    const int last_pos = position3d(hist_size-1,y,z,hist_size,y_size);

    for (int g_pos = first_pos+1; g_pos <= last_pos-1; g_pos+=1) {
        const bool ripple = ripple_away(in_class[cur_pos-1], in_class[cur_pos], in_class[cur_pos+1]);
        if (ripple) {
            in_out_class[g_pos] = 5;
        }
    }
    
    return;
}

bool pseudo_flat_2(int prev_prev_class, int prev_class, int cur_class, int next_class) {

    if (prev_class == 9 && cur_class == 1 || prev_class == 1 && cur_class == 9) {
            if (5 <= prev_prev_class && prev_prev_class <= 8 && 2 <= next_class && next_class <= 5)
                return true;                
            if (2 <= prev_prev_class && prev_prev_class <= 5 && 5 <= next_class && next_class <= 8)
                return true;
    }
    return false;
}

bool pseudo_flat_4(int prev_prev_class, int prev_class, int cur_class, int next_class) {
    if (prev_prev_class == 9 && 9 == next_class && 5 <= prev_class && prev_class <= 8 && 2 <= cur_class && cur_class <= 5)
        return true; //classes[index-2:index+1] = 5
    if (prev_prev_class == 1 && 1 == next_class && 2 <= prev_class && prev_class <= 5 && 5 <= cur_class && cur_class <= 8)
        return true; //classes[index-2:index+1] = 5
    return false;
}

kernel ip_flat_kernel(global int *in_out_class, global int* skip, int hist_size) {
    const int y = get_global_id(0);
    const int z = get_global_id(1);
    //const int z = get_global_id(2);
    const int y_size = get_global_size(0);
    const int z_size = get_global_size(1);
    //const int z_size = get_global_size(2);
    const int skip_pos = position2d(y, z, y_size);

    if (z < 2 || z > z_size-1 || skip[skip_pos] == 1){
        return;
    }

    const int first_pos = position3d(0,y,z,hist_size,y_size);
    const int last_pos = position3d(hist_size-1,y,z,hist_size,y_size);
    
    for (int g_pos = first_pos+2; g_pos <= last_pos-1; g_pos+=1) {
        bool pf2 = pseudo_flat_2(in_out_class[g_pos-2], in_out_class[g_pos-1], in_out_class[g_pos], in_out_class[g_pos+1]);
        if(pf2) {
            in_out_class[g_pos-1] = 5;
            in_out_class[g_pos] = 5;            
        }
        bool pf4 = pseudo_flat_4(in_out_class[g_pos-2], in_out_class[g_pos-1], in_out_class[g_pos], in_out_class[g_pos+1]);
        if(pf4) {
            in_out_class[g_pos-2] = 5;
            in_out_class[g_pos-1] = 5;
            in_out_class[g_pos] = 5;            
            in_out_class[g_pos+1] = 5;
        }
    }
    return;    
}

/*
This can run in parallel on each hist bin with ignoring first and last! --> calculate index first!
*/
int classify(float prev, float cur, float next){
    if (prev == cur && cur == next)
        return 5;// # flat

    if (prev > cur && cur < next)
        return 1; //# min
    if (prev < cur && cur > next)
        return 9; //# max

    const float slope_prev_cur = cur-prev;
    const float slope_cur_next = next-cur;

    if (prev <= cur && cur <= next){ //# up
        if (slope_prev_cur < slope_cur_next)
            return 2; // # left
        if (slope_prev_cur == slope_cur_next)
            return 3; // # wp
        if (slope_prev_cur > slope_cur_next)
            return 4; // # right
    }

    if (prev >= cur && cur >= next){ //# down
        if (slope_prev_cur < slope_cur_next)
            return 6; // # left
        if (slope_prev_cur == slope_cur_next)
            return 7; // # wp
        if (slope_prev_cur > slope_cur_next)
            return 8; # right
    }
    return 0;
}

kernel classify_array_kernel(global int *in_hist, global int *out_class, global int* skip) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);
    const int x_size = get_global_size(0);
    const int y_size = get_global_size(1);
    const int z_size = get_global_size(2);

    const int cur_pos = position3d(x,y,z,x_size,y_size);
    const int skip_pos = position2d(y, z, y_size);
    int hist_class;

    if (z < 1 || z > z_size-1 || skip[skip_pos] == 1){
        hist_class = 0;
    } else {
        hist_class = classify(in_class[cur_pos-1], in_class[cur_pos], in_class[cur_pos+1]);
    }
    out_class[cur_pos] = smooth_class;
}


int find_night_fog_thr(int *hist, int peak_bin, int right_wp_bin, int right_min_bin){
    const int steps_after_thr_to_break = 2;
    const float max_slope_ratio = 0.7;

    float wp_to_min_slope = slope(right_wp_bin, right_min_bin, hist[right_wp_bin], hist[right_min_bin]);
    float peak_to_min_slope = slope(peak_bin, right_min_bin, hist[peak_bin], hist[right_min_bin]);
    float min_slope = min(wp_to_min_slope,peak_to_min_slope); // # must be <= 0 as the peak must be above the min

    bool reached_min_slope = false;
    int thr_bin = peak_bin;
    float thr_slope = 0;

    for (int index = peak_bin; index <= right_min_bin; index+=1){
        float thr_index_slope = slope(thr_bin, index, hist[thr_bin], hist[index]);
        if( (!reached_min_slope || thr_index_slope <= min_slope) && thr_index_slope <= thr_slope * max_slope_ratio){
            thr_bin = index;
            thr_slope = thr_index_slope;

            if ((!reached_min_slope) && thr_index_slope <= min_slope)
                reached_min_slope = true;
        }

        if (reached_min_slope && index >= thr_bin + steps_after_thr_to_break)
            break;
    }
    return thr_bin;
}


int find_cloud_thr(int *hist, int left_min_bin, int left_wp_bin, int peak_bin):
    const steps_after_thr_to_break = 2;
    float max_slope_ratio = 0.7;

    float wp_to_min_slope = slope(left_min_bin, left_wp_bin, hist[left_min_bin], hist[left_wp_bin]);
    float peak_to_min_slope = slope(peak_bin, left_min_bin, hist[peak_bin], hist[left_min_bin]);
    float max_slope = max(wp_to_min_slope,peak_to_min_slope); // # must be >= 0 as the peak must be above the min

    bool reached_max_slope = false;
    int thr_bin = peak_bin;
    float thr_slope = 0;

    for(int index = peak_bin; index >= left_min_bin, index-=1){

        float thr_index_slope = slope(thr_bin, index, hist[thr_bin], hist[index]);
        if( (!reached_max_slope || thr_index_slope >= max_slope) && thr_index_slope >= thr_slope * max_slope_ratio){
            thr_bin = index;
            thr_slope = thr_index_slope;

            if ((!reached_max_slope) && thr_index_slope >= max_slope)
                reached_max_slope = true;

        if (reached_max_slope and index <= thr_bin - steps_after_thr_to_break)
            break;
    }
    return thr_bin;
}


kernel void find_all_the_things8_kernel(global int *in_hist, global int *in_classes, global int *out_bin_ids, int hist_range, int hist_bin_width, int min_bin_zero_based, int last_bin_zero_based, int max_bin_zero_based, int zero_bin_zero_based) {
    // classes = pseudo_flat_away(ripple_away(classify_hist(hist))) #0: unknown, 1: mi, 2:left inc, 3: left str, 4: left dec, 5: flat, 6: right inc, 7: right str, 8: right dec, 9: max

//    int min_bin = 0;
//    int last_bin = 20;
//    int max_bin = len(hist)-3;
//    float zero_bin = hist_value_to_bin(0, hist_range=hist_range, hist_bin_width=hist_bin_width);

    const int y = get_global_id(0); //"x"
    const int z = get_global_id(1); //"y"    
    const int y_size = get_global_size(0);
    const int z_size = get_global_size(1);
    
    const int flat_pos = position2d(y, z, y_size);

    if (z < 2 || z > z_size-1 || skip[flat_pos] == 1){
        return;
    }

    const int _first_pos = position3d(0,y,z,hist_size,y_size);
    const int _last_pos = position3d(hist_size-1,y,z,hist_size,y_size);
    

    const int _off = first_pos;
    const int _small_value = _off; // 0
    const int max_bin = _off + max_bin_zero_based;
    const int min_bin = _off + min_bin_zero_based;
    const int zero_bin = _off + zero_bin_zero_based;
    const int last_bin = _off + last_bin_zero_based;

//    # init from run to the right, next is at max
//    # small drops
    int sd_right_min_bin =  max_bin;
    int sd_right_wp_bin =   max_bin;
    int sd_peak_bin =       max_bin;
    int sd_left_wp_bin =    max_bin;
    int sd_left_min_bin =   max_bin;
    int sd_right_wp_slope = _small_value;

//    # large drops
    int ld_right_min_bin =  min_bin;
    int ld_right_wp_bin =   min_bin;
    int ld_peak_bin =       min_bin;
    int ld_left_wp_bin =    min_bin;
    int ld_left_min_bin =   min_bin;
    int ld_left_wp_slope = _small_value;

//    ## moving peaks
//    # next is at max
    int next_right_min_bin = max_bin;
    int next_right_wp_bin =  max_bin;
    int next_peak_bin =      max_bin;
    int next_left_wp_bin =   max_bin;
    int next_left_min_bin =  max_bin;

//    # current is at min with exception of the right min!
    int cur_right_min_bin = max_bin;
    int cur_right_wp_bin =  max_bin;
    int cur_peak_bin =      min_bin;
    int cur_left_wp_bin =   min_bin;
    int cur_left_min_bin =  min_bin;

    int prev_right_min_bin = min_bin;
    int prev_right_wp_bin =  min_bin;
    int prev_peak_bin =      min_bin;
    int prev_left_wp_bin =   min_bin;
    int prev_left_min_bin =  min_bin;

//    #init slopes
    int next_right_wp_slope = _small_value;
    int next_left_wp_slope = next_right_wp_slope;
    int cur_right_wp_slope = next_left_wp_slope;
    int cur_left_wp_slope = cur_right_wp_slope;
    int prev_right_wp_slope = cur_left_wp_slope;


//    # start from the right start bin "sweep_right_start_bin". ths might be a zero value!
//    for step, index in enumerate(xrange(next_right_min_bin, min_bin, -1), start=last_bin):
    int step = 0;
    for (int index = last_bin; index >= min_bin; index-=1) {
        step += 1;

//        # some other things
//        #print "step:", step, "index:", index, "class:", classes[index]

//        # right min
//        if cur_peak_bin < index <= next_left_wp_bin:
        if (cur_peak_bin < index && index <= next_left_wp_bin){
//            #print " ~ 1 cur_right_min_bin"
//            if 1 <= classes[index] <= 5:
            if (1 <= classes[index] && classes[index] <= 5){
//                #print "  > cur_right_min_bin"
                cur_right_wp_bin = cur_right_min_bin = index;
//                # move all the others back to the min_bin

                cur_peak_bin = cur_left_wp_bin = cur_left_min_bin = min_bin; // # slope
                cur_right_wp_slope = cur_left_wp_slope = prev_right_wp_slope = 0.0;
            }
        }

//        # right wp
//        if cur_left_wp_bin < index <= cur_right_min_bin:
        if (cur_left_wp_bin < index  && index <= cur_right_min_bin){
//            #print " ~ 2 cur_right_wp_bin"
            index_to_right_wp_slope = slope(index, cur_right_wp_bin, hist[index], hist[cur_right_wp_bin]);
//            if (classes[index] == 1 or 5 <= classes[index] <= 7) and index_to_right_wp_slope <= cur_right_wp_slope:
            if ((classes[index] == 1 || 5 <= classes[index] <= 7) && index_to_right_wp_slope <= cur_right_wp_slope){
//                #print "  > cur_right_wp_bin -> cur_right_wp_slope:", cur_right_wp_slope, " <= index_to_right_wp_slope", index_to_right_wp_slope
                cur_right_wp_bin = index; //# slope
                cur_right_wp_slope = index_to_right_wp_slope;

                cur_peak_bin = cur_left_wp_bin = cur_left_min_bin = prev_right_wp_bin = prev_peak_bin = min_bin; // # slope
                cur_left_wp_slope = prev_right_wp_slope = 0;
            }
        }

//        # peak
//        if cur_left_min_bin < index <= cur_right_wp_bin:
        if (cur_left_min_bin < index && index <= cur_right_wp_bin){
//            #print " ~ 3 cur_peak_bin"
//            if classes[index] == 9 or (2 <= classes[index-1] <= 5 and 6 <= classes[index] <= 8):  #and hist[index] > hist[cur_peak_bin]:
            if (classes[index] == 9 || (2 <= classes[index-1] && classes[index-1] <= 5 && 6 <= classes[index] && classes[index] <= 8) {
//                #print "  > cur_peak_bin"
                cur_left_wp_bin = cur_peak_bin = index;

                cur_left_min_bin = prev_right_wp_bin = prev_peak_bin = min_bin; // # slope
                cur_left_wp_slope = prev_right_wp_slope = 0.0;
            }
        }

//        # left wp
//        if prev_right_wp_bin < index <= cur_peak_bin:
        if (prev_right_wp_bin < index && index <= cur_peak_bin) {
//            #print " ~ 4 cur_left_wp_bin"
            index_to_cur_left_wp_slope = slope(index, cur_left_wp_bin, hist[index], hist[cur_left_wp_bin]);
//            if (classes[index] == 9 or 2 <= classes[index] <= 5) and index_to_cur_left_wp_slope >= cur_left_wp_slope:
            if ((classes[index] == 9 || 2 <= classes[index] && classes[index] <= 5) && index_to_cur_left_wp_slope >= cur_left_wp_slope){
//                #print "  > cur_left_wp_bin -> cur_left_wp_slope:", cur_left_wp_slope, " <= index_to_cur_peak_slope", index_to_cur_left_wp_slope
                cur_left_wp_slope = index_to_cur_left_wp_slope;
                cur_left_wp_bin = index;

                cur_left_min_bin = prev_right_wp_bin = min_bin; // # slope
                prev_right_wp_slope = 0.0;
            }
        }

//        #cur left min + prev_right_min
//        if prev_right_wp_bin < index <= cur_left_wp_bin:
        if (prev_right_wp_bin < index && index <= cur_left_wp_bin){
//            #print " ~ 5 cur_left_min_bin |OR| prev_right_min_bin"
//            if classes[index] == 1 or (5 <= classes[index-1] <= 8 and 2 <= classes[index] <= 4):
            if (classes[index] == 1 || (5 <= classes[index-1] && classes[index-1] <= 8 && 2 <= classes[index] && classes[index] <= 4)) {
//                #print "  > 5.1 cur_left_min_bin"
                prev_right_min_bin = cur_left_min_bin = index;
                prev_right_wp_bin = prev_peak_bin = min_bin; // # slope
                prev_right_wp_slope = 0.0;
            }

//            if classes[index] == 1 or (6 <= classes[index-1] <= 8 and 2 <= classes[index] <= 5):
            if (classes[index] == 1 || (6 <= classes[index-1] && classes[index-1] <= 8 && 2 <= classes[index] && classes[index] <= 5)) {
//                #print "  > 5.2 prev_right_min_bin"
                prev_right_min_bin = index;
                prev_right_wp_bin = prev_peak_bin = min_bin; //# slope
                prev_right_wp_slope = 0.0;
            }

//        if (prev_left_wp_bin < index <= cur_left_wp_bin and (6 <= classes[index-1] <= 9)) or index == last_bin:
        if (prev_left_wp_bin < index && index <= cur_left_wp_bin && (6 <= classes[index-1] && classes[index-1] <= 9)) || index == last_bin) {
//            #print "there is something new!"

            // # is this the small droplet peak? -> skip it
            // #if zero_bin < cur_peak_bin: #Todo:maybe we need a better metric later...
            //    #print "  ? zero_bin: ", zero_bin, " < ", cur_peak_bin, " :cur_peak_bin"
            if (cur_peak_bin <= zero_bin){
                //# is this the steepest right flank? -> set it as new LD flank
                if cur_right_wp_slope < sd_right_wp_slope {
                    //#print "  ? cur_right_wp_slope: ", cur_right_wp_slope, " < ",sd_right_wp_slope, " :sd_right_wp_slope"
                    ld_right_min_bin = sd_right_min_bin = cur_right_min_bin;
                    ld_right_wp_bin = sd_right_wp_bin = cur_right_wp_bin;
                    ld_peak_bin = sd_peak_bin = cur_peak_bin;
                    ld_left_wp_bin = sd_left_wp_bin = cur_left_wp_bin;
                    ld_left_min_bin = sd_left_min_bin = cur_left_min_bin;
                    sd_right_wp_slope = cur_right_wp_slope;
                    ld_left_wp_slope = cur_left_wp_slope;
                 } else {
                    cur_peak_ld_peak_slope = slope(cur_peak_bin, ld_peak_bin, hist[cur_peak_bin], hist[ld_peak_bin]);                    
                    if (cur_left_wp_slope > ld_left_wp_slope || ld_left_wp_slope * 0.7 <= cur_left_wp_slope && cur_left_wp_slope >= cur_peak_ld_peak_slope) && cur_peak_bin > (ld_peak_bin - ((max_bin - min_bin)/4)) && 30 <= cur_peak_bin){
                        ld_right_min_bin  = cur_right_min_bin;
                        ld_right_wp_bin = cur_right_wp_bin;
                        ld_peak_bin = cur_peak_bin;
                        ld_left_wp_bin = cur_left_wp_bin;
                        ld_left_min_bin = cur_left_min_bin;
                        ld_left_wp_slope = cur_left_wp_slope;
                    }
                }
            }

//            #next_right_min_bin = cur_right_min_bin
//            #next_right_wp_bin = cur_right_wp_bin
//            #next_peak_bin = cur_peak_bin
            next_left_wp_bin = cur_left_wp_bin;
//            #next_left_min_bin = cur_left_min_bin
//            #next_left_wp_slope = cur_left_wp_slope
//            #next_right_wp_slope = cur_right_wp_slope
//            # push everything back:
            cur_right_min_bin = prev_right_min_bin;
            cur_right_wp_bin = index;
            cur_peak_bin = cur_left_wp_bin = cur_left_min_bin = min_bin;
            cur_right_wp_slope = cur_left_wp_slope = 0;
    }

    //return [ld_left_min_bin, ld_left_wp_bin, ld_peak_bin, ld_right_wp_bin, ld_right_min_bin, sd_left_min_bin, sd_left_wp_bin, sd_peak_bin, sd_right_wp_bin, sd_right_min_bin];
    // TODO: 1 find threhsold or return something!!!
}
