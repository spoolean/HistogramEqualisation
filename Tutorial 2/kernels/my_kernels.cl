// OpenCL kernel to convert an RGB image to grey scale.
kernel void rgb2grey(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);
	int image_size = get_global_size(0) / 3; //each image consists of 3 colour channels
	int colour_channel = id / image_size; // 0 - red, 1 - green, 2 - blue
	int value;

	if (colour_channel == 0) {
		// Get the grey value of this coloured value by averaging the red, green and blue values.
		value = A[id] * 0.2126 + A[id + image_size] * 0.71526 + A[id + (image_size * 2)] * 0.0722;
		// Copy the value across all three colour chanels.
		B[id] = value;
		B[id + image_size] = value;
		B[id + (image_size * 2)] = value;
	}
}

// A simple OpenCL kernel which copies all pixels from A to B.
kernel void identity(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);
	B[id] = A[id];
}

// OpenCl kernel which calculates an intensity histogram for a given image.
// This kernel is serial and slow. 
kernel void histogram(global const uchar* A, global int* B) {
	int id = get_global_id(0);
	int value = A[id];
	atomic_inc(&B[value]);
}

// OpenCL kernel which calculates an intensity histogram using local memory.
// This kernel is much faster than the above because it calculates multiple local histograms across the device,
// and combines them at the end into a single global histogram. This greatly reduces the proccessing time because,
// you do not need to lock and unlock each bin on the global memory (which is slow).
kernel void local_global(global const uchar* A, global int* H, local int* LH, int A_size, int histBins) {
	int gid = get_global_id(0); 
	int lid = get_local_id(0);
	int lsize = get_local_size(0);
	int gsize = get_global_size(0);
	
	// Set Local Histogram Bins to 0
	for (int i = lid; i < histBins; i += lsize)
	{
		LH[i] = 0;
	}

	// Wait for all threads to finish setting local histogram bins to 0
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// Compute Local Histogram
	for (int i = gid; i < A_size; i += gsize)
	{
		atomic_inc(&LH[A[i]]);
	}
	
	// Wait for all threads to finish computing local histogram
	barrier(CLK_LOCAL_MEM_FENCE);

	// Copy Local Histograms to Global Histogram
	for (int i = lid; i < histBins; i += lsize)
	{
		atomic_add(&H[i], LH[i]);
	}
}

// OpenCl kernel which calulates the cumulative histogram from the intensity histogram.
// This kernal uses the Hillis-Steel Inclusive parralel algorithm. This algortihm,
// works on global memory but is fast because it does not need to write to a bin more than once,
// so multiple work items can be performed at once. This cumulative histogram had to be inclusive,
// so that no intensity values were lost. Moreover, this algorithm is suited to this role as there is more Proccessors than work items (256).
kernel void cumulativeHistogram(global int* A, global int* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	global int* C;

	for (int stride = 1; stride < N; stride *= 2) {
		B[id] = A[id];
		if (id >= stride)
			B[id] = A[id] + A[id - stride];

		barrier(CLK_GLOBAL_MEM_FENCE); // sync the step

		C = A; A = B; B = C; // swap A & B between steps
	}
}

// OpenCl kernel which normalises the cumulative histogram to a maximum value of 255. 
// Take a ratio of the actual value in relation to a maximum 255.
kernel void normalise(global int* A, global int* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	int value = A[id];
	// Normalise the histogram to a maximum of 255.
	B[id] = value * (double)255 / A[255];
}

// OpenCl kernel which uses the cumalative histogram as a lookup table for the original intensities
kernel void lookup(global const uchar* A, global const int* B, global uchar* C) {
	int id = get_global_id(0);
	int value = A[id]; // Take the original value. 
	int lookup_value = B[value]; // Use the orginial intesity as a lookup value in the normalised histogram.
	C[id] = lookup_value;// Copy the lookup value to the output image.
}
