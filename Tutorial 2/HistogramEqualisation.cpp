// Parrallel Programming CMP3752M James Roche 19704410
// Assignment 1
// This assigment was completed with the use of Tutorial 2 as a base and the kernels from Tutorial 3.
// The code below can be modified by commenting and uncommenting the section related to the histogram algorithm. This can be used to changed it from serial to parrallel.
// The histogram bin number hist, can be modified. Any value smaller than 256 will make the image darker and darker until it becomes black. Any value larger than 256 will result in an error, because the image cannot represent values larger than 255. 
// Blelloch and Hillis-Steel have been implemented and can be commented and uncommented to allow the use of each. 
// The parrallel histogram algorithm atomic histogram is a modification of the code provided in the lectures (the lecture code doesn't work) this algorithm is over three times faster than the simple histogram because of it's use of local memory that is much faster. 
// When used on the test_large file the time for the simple histogram was 31679520ns to complete the kernel, the time for the atomic local histogram: 9825600ns. This resulted in the overall time for all the kernels to be completed to be reduced by half. From 4secs to 2secs. 
// Aditional features that have been included are the abbility to change the number of bins from 256 -> 128 and lower, the ability to swap the histogram calulation algortihm, the abillity to swap multiple scan patterns and the abilty to use colour images. 
// Moreover, each algorithm used, bar simple histogram, makes use of parrallel features. Finally, each of the kernels proccesing times are relayed to the user in the console, and the histogram equlisation of the image is correct. 


#include <iostream>
#include <vector>

#include "Utils.h"
#include "CImg.h"

using namespace cimg_library;

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -f : input image file (default: test.ppm)" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
	
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	string image_filename = "colour_test.ppm";

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	cimg::exception_mode(0);

	//detect any potential exceptions
	try {
		bool histFired = false;
		bool hillisFired = false;
		CImg<unsigned char> image_input(image_filename.c_str());
		CImgDisplay disp_input(image_input,"input");

		//Part 2 - host operations
		//2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Runing on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		// Create a queue to which we will push commands for the device and enable profiling. 
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);
		
		// Events used to monitor the resources for each kernel. 
		cl::Event rgbEvent;
		cl::Event greyEvent;
		cl::Event histEvent;
		cl::Event atomicHistEvent;
		cl::Event cumulativeHistEvent;
		cl::Event blellochCumulEvent;
		cl::Event normaliseHistEvent;
		cl::Event mapHistEvent;

		//2.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernels/my_kernels.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try { 
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		// Part 3 Memory Allocation
		
		// 3.1 Host Memory Allocation
		typedef int mytype;

		// This value can be changed but must be a multiple of 256 and not higher that 256 (An image cannot be represented by values higher than 255). 
		int hist = 4;
		
		// Vectors
		vector<int> histogram(hist);
		vector<unsigned char> intensity_map(image_input.size()); // Final output for image reproduction.
		vector<int> binvals(hist);
		
		// The memory allocation size for the histogram. It has to be in bytes
		size_t histogramSize = histogram.size() * sizeof(mytype);

		// 3.2 Device - buffers
		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_input.size()); // The input image
		cl::Buffer initialImageArray(context, CL_MEM_READ_WRITE, image_input.size()); // The image after being changed to greyscale
		cl::Buffer intensityHistogram(context, CL_MEM_READ_WRITE, histogramSize); // The intensity histogram calculated from the image 
		cl::Buffer cumulativeHistogram(context, CL_MEM_READ_WRITE, histogramSize); // The cumulative histogram
		cl::Buffer normalisedHistogram(context, CL_MEM_READ_WRITE, histogramSize); // The normalisation of the cumulative histogram
		cl::Buffer intensityMap(context, CL_MEM_READ_WRITE, image_input.size()); // The output after using the normalised histogram as a LUT
		cl::Buffer binsizeBuffer(context, CL_MEM_READ_WRITE, histogramSize); // The binsize buffer

		int increments = 256 / hist;
		for (size_t i = 0; i < hist; i++)
		{
			binvals[i] = i * increments;
		}
		 
		// 3.3 Copy image to device memory
		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);
		queue.enqueueWriteBuffer(binsizeBuffer, CL_TRUE, 0, histogramSize, & binvals[0]);
		queue.enqueueFillBuffer(intensityHistogram, 0, 0, histogramSize);

		// 4 Setup and execute the kernels (i.e. device code)
		
		// 4.1 Firstly, change the image to greyscale so the intensites can be counted
		// Check if dev_image_input is RGB.
		if (image_input.spectrum() == 3) {
			// If RGB, convert to grayscale.
			cl::Kernel kernel_rgb2gray(program, "rgb2grey");
			kernel_rgb2gray.setArg(0, dev_image_input);
			kernel_rgb2gray.setArg(1, initialImageArray);
			queue.enqueueNDRangeKernel(kernel_rgb2gray, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &rgbEvent);
		}
		else {
			// If grayscale, just copy into initialImageArray Buffer.
			cl::Kernel kernel_copy(program, "identity");
			kernel_copy.setArg(0, dev_image_input);
			kernel_copy.setArg(1, initialImageArray);
			queue.enqueueNDRangeKernel(kernel_copy, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &greyEvent);
		}
		
		// 4.2 Calculation of the histogram.
		// The first kernel that is commented is a serial version, please comment and uncomment this one, 
		// and the atomic_histogram to see the difference in parrellelisation.

		// Calculate an intensity histogram using the atomic_inc method.
		// This method is slow and serial as the bins have to be locked and,
		// unlocked sequentially per increment.
		histFired = true;
		cl::Kernel kernel_histogram(program, "histogram");
		kernel_histogram.setArg(0, initialImageArray);
		kernel_histogram.setArg(1, intensityHistogram);
		kernel_histogram.setArg(2, hist);
		kernel_histogram.setArg(3, (int)image_input.size());
		kernel_histogram.setArg(4, binsizeBuffer);
		queue.enqueueNDRangeKernel(kernel_histogram, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &histEvent);
		// Read to console 
		queue.enqueueReadBuffer(intensityHistogram, CL_TRUE, 0, histogramSize, &histogram[0]);
		cout << "Histogram = " << histogram << endl << endl;


		//// Calculate the intensity histogram using a parrallel method with local memory, and local to global reductions. 
		//// This method is much faster than the serial version, as it does not need, 
		//// to lock and unlock the global bins. Instead it uses a local memory buffer to store the local histograms.
		//// This way, only the local bins are locked and unlocked, and the global bins are only locked once to add the local,
		//// histograms together. This algoirthm is 3x faster than the serial version when tested on the large_test image. */
		//cl::Kernel kernel_atomic_histogram(program, "local_global");
		//kernel_atomic_histogram.setArg(0, initialImageArray);
		//kernel_atomic_histogram.setArg(1, intensityHistogram);
		//kernel_atomic_histogram.setArg(2, cl::Local(histogramSize));
		//kernel_atomic_histogram.setArg(3, (int)image_input.size());
		//kernel_atomic_histogram.setArg(4, hist);
		//kernel_atomic_histogram.setArg(5, binsizeBuffer);
		//queue.enqueueNDRangeKernel(kernel_atomic_histogram, cl::NullRange, cl::NDRange(image_input.size()), cl::NDRange(histogram.size()), NULL, &atomicHistEvent);
		//// Read to console
		//queue.enqueueReadBuffer(intensityHistogram, CL_TRUE, 0, histogramSize, &histogram[0]);
		//cout << "Histogram = " << histogram << endl << endl;


		// Calculate a cumulative histogram of the intensity histogram. 
		// An inclusive Hillis-Steel scan pattern which keeps the individual parts, 
		// as it moves along the array. This one can also be commented out to allow for the blelloch scan below,
		// comment and uncomment as apropriate. 
		hillisFired = true;
		cl::Kernel kernel_cumulativeHistogram(program, "cumulativeHistogram");
		kernel_cumulativeHistogram.setArg(0, intensityHistogram);
		kernel_cumulativeHistogram.setArg(1, cumulativeHistogram);
		kernel_cumulativeHistogram.setArg(2, cl::Local(histogramSize));
		kernel_cumulativeHistogram.setArg(3, cl::Local(histogramSize));
		queue.enqueueNDRangeKernel(kernel_cumulativeHistogram, cl::NullRange, cl::NDRange(histogramSize), cl::NDRange(histogram.size()), NULL, &cumulativeHistEvent);
		// Read to console
		queue.enqueueReadBuffer(cumulativeHistogram, CL_TRUE, 0, histogramSize, &histogram[0]);
		cout << "Hillis-Steel Cumulative Histogram = " << histogram << endl << endl;
		
		//// A Blelloch exclusive scan pattern which keeps the individual parts,
		//// but they are stored in the wrong bin (exclusive). This will make the image appear brighter.
		//// This one can also be commented out to allow for the inclusive Hillis-Steel scan above,
		//// comment and uncomment as apropriate.
		//cl::Kernel blellochCumuHistogram(program, "blellochCumulative");
		//blellochCumuHistogram.setArg(0, intensityHistogram);
		//blellochCumuHistogram.setArg(1, cumulativeHistogram);
		//queue.enqueueNDRangeKernel(blellochCumuHistogram, cl::NullRange, cl::NDRange(histogramSize), cl::NullRange, NULL, &blellochCumulEvent);
		//// Read to console
		//queue.enqueueReadBuffer(cumulativeHistogram, CL_TRUE, 0, histogramSize, &histogram[0]);
		//cout << "Blelloch Cumulative Histogram = " << histogram << endl;
		
		
		// Normalise the cumlative histogram to a maximum value of 255.
		cl::Kernel kernel_normaliseHistogram(program, "normalise");
		kernel_normaliseHistogram.setArg(0, cumulativeHistogram);
		kernel_normaliseHistogram.setArg(1, normalisedHistogram);
		kernel_normaliseHistogram.setArg(2, hist);
		queue.enqueueNDRangeKernel(kernel_normaliseHistogram, cl::NullRange, cl::NDRange(256), cl::NullRange, NULL, &normaliseHistEvent);
		// Read to console
		queue.enqueueReadBuffer(normalisedHistogram, CL_TRUE, 0, histogramSize, &histogram[0]);
		cout << "Normalised Histogram = " << histogram << endl << endl;

		// Use the cumulative histogram as a lookup table to map the intensity values to the original image.
		cl::Kernel kernel_lookup(program, "lookup");
		kernel_lookup.setArg(0, dev_image_input);
		kernel_lookup.setArg(1, normalisedHistogram);
		kernel_lookup.setArg(2, intensityMap);
		kernel_lookup.setArg(3, hist);
		kernel_lookup.setArg(4, binsizeBuffer);
		queue.enqueueNDRangeKernel(kernel_lookup, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &mapHistEvent);
		
		
		// 4.3 Copy the result from device to the host.
		queue.enqueueReadBuffer(intensityMap, CL_TRUE, 0, intensity_map.size(), &intensity_map.data()[0]);

		CImg<unsigned char> output_image(intensity_map.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
		CImgDisplay disp_output(output_image,"output");
		
		
		// 4.4 Timings of the events for each of the kernels.
		// The events are not used in the final version, but can be used to see the time taken for each kernel.
		// The events are also used to see the time taken for the entire process.
		if (image_input.spectrum() == 3) {
			// If the image had to be converted to RGB then cout how long it took.
			std::cout << "RGB to greyscale took: " << rgbEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - rgbEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << "ns to complete" << std::endl;
		}
		else
		{
			// If the image was already greyscale, then cout how long it took.
			std::cout << "Greyscale copy took: " << greyEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - greyEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << "ns to complete" << std::endl;
		}
		
		if (histFired)
		{
			// If the histogram was calculated, then cout how long it took.
			std::cout << "Histogram took: " << histEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - histEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << "ns to complete" << std::endl;
		}
		else
		{
			// If the atomic histogram was calculated, then cout how long it took.
			std::cout << "Atomic histogram took: " << atomicHistEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - atomicHistEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << "ns to complete" << std::endl;
		}
		
		if (hillisFired)
		{
			// If the Hillis-Steel cumulative histogram was calculated, then cout how long it took.
			std::cout << "Hillis-Steel optimized cumulative histogram took: " << cumulativeHistEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - cumulativeHistEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << "ns to complete" << std::endl;
		}
		else
		{
			// If the blelloch cumulative histogram was calculated, then cout how long it took.
			std::cout << "Blelloch cumulative histogram took: " << blellochCumulEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - blellochCumulEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << "ns to complete" << std::endl;
		}
		
		std::cout << "Normalise histogram took: " << normaliseHistEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - normaliseHistEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << "ns to complete" << std::endl;
		std::cout << "Lookup table took: " << mapHistEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - mapHistEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << "ns to complete" << std::endl;
		
		// Add all start and end times together to get the total time.
		cl_ulong commandStart = (image_input.spectrum() == 3) ? rgbEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() : greyEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		std::cout << "Total time for the kernels to execute from start to finish was: " <<  (float)(mapHistEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>()-commandStart)/10000000 << "s to complete" << std::endl;

 		while (!disp_input.is_closed() && !disp_output.is_closed()
			&& !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
		    disp_input.wait(1);
		    disp_output.wait(1);
	    }		

	}
	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}

	return 0;
}
