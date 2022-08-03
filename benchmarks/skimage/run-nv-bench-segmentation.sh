#!/bin/bash
param_shape=(512,512 3840,2160 3840,2160,3 192,192,192)

# these require an integer-valued label image
param_filt=(clear_border expand_labels relabel_sequential find_boundaries mark_boundaries random_walker)
param_dt=(float32)
param_dt_label=(uint8 uint32)
for shape in "${param_shape[@]}"; do
    for filt in "${param_filt[@]}"; do
        for dt in "${param_dt[@]}"; do
            for dt_label in "${param_dt_label[@]}"; do
                python cucim_segmentation_bench.py -f $filt -i $shape -d $dt --dtype_label $dt_label -t 10
            done
        done
    done
done

# these do not require an integer-valued input image
param_filt=(inverse_gaussian_gradient morphological_geodesic_active_contour morphological_chan_vese)
param_dt=(float32)
for shape in "${param_shape[@]}"; do
    for filt in "${param_filt[@]}"; do
        for dt in "${param_dt[@]}"; do
            python cucim_segmentation_bench.py -f $filt -i $shape -d $dt -t 10
        done
    done
done
