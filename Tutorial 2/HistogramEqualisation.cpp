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
	string image_filename = "colour.ppm";

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
		CImg<unsigned char> image_input(image_filename.c_str());
		CImgDisplay disp_input(image_input,"input");

		//Part 2 - host operations
		//2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Runing on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context);

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
		
		typedef int mytype;
		
		//Vectors
		vector<unsigned char> output_buffer(image_input.size());
		vector<int> histogram(256);
		vector<int> cumulative_histogram(256);
		vector<unsigned char> intensity_map(image_input.size());
		
		size_t histogramSize = histogram.size() * sizeof(mytype);

		////Part 3 - memory allocation
		////host - input
		//// input vector A with 10 random values from 0-9
		//std::vector<int> A(10);

		////the following part adjusts the length of the input vector so it can be run for a specific workgroup size
		////if the total input length is divisible by the workgroup size
		////this makes the code more efficient
		//size_t local_size = 10;

		//size_t padding_size = A.size() % local_size;

		////if the input vector is not a multiple of the local_size
		////insert additional neutral elements (0 for addition) so that the total will not be affected
		//if (padding_size) {
		//	//create an extra vector with neutral values
		//	std::vector<int> A_ext(local_size - padding_size, 0);
		//	//append that extra vector to our input
		//	A.insert(A.end(), A_ext.begin(), A_ext.end());
		//}

		//size_t input_elements = A.size();//number of input elements
		//size_t input_size = A.size() * sizeof(mytype);//size in bytes
		//size_t nr_groups = input_elements / local_size;

		//Part 4 - device operations

		//device - buffers
		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_input.size());
		cl::Buffer dev_image_output(context, CL_MEM_READ_WRITE, image_input.size());
		
		cl::Buffer initialImageArray(context, CL_MEM_READ_WRITE, image_input.size());
		cl::Buffer intensityHistogram(context, CL_MEM_READ_WRITE, histogramSize);
		cl::Buffer cumulativeHistogram(context, CL_MEM_READ_WRITE, histogramSize);
		cl::Buffer normalisedHistogram(context, CL_MEM_READ_WRITE, histogramSize);
		cl::Buffer intensityMap(context, CL_MEM_READ_WRITE, image_input.size());
		 
		//4.1 Copy images to device memory
		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);
		queue.enqueueFillBuffer(intensityHistogram, 0, 0, histogramSize);

		// 4.2 Setup and execute the kernels (i.e. device code)
		
		// If the image is RGB then there is more inforamtion then there are more proccesses that need to be done 
		// on the array. 
		// Check if dev_image_input is RGB.
		if (image_input.spectrum() == 3) {
			//if RGB, convert to grayscale
			cl::Kernel kernel_rgb2gray(program, "rgb2grey");
			kernel_rgb2gray.setArg(0, dev_image_input);
			kernel_rgb2gray.setArg(1, initialImageArray);
			queue.enqueueNDRangeKernel(kernel_rgb2gray, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange);
		}
		else {
			//if grayscale, just copy
			cl::Kernel kernel_copy(program, "identity");
			kernel_copy.setArg(0, dev_image_input);
			kernel_copy.setArg(1, initialImageArray);
			queue.enqueueNDRangeKernel(kernel_copy, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange);
		}
		
		// Calculate an intensity histogram of the initialImageArray
		cl::Kernel kernel_histogram(program, "histogram");
		kernel_histogram.setArg(0, initialImageArray);
		kernel_histogram.setArg(1, intensityHistogram);
		queue.enqueueNDRangeKernel(kernel_histogram, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange);
		
		// Calculate a cumulative histogram of the intensity histogram
		cl::Kernel kernel_cumulativeHistogram(program, "cumulativeHistogram");
		kernel_cumulativeHistogram.setArg(0, intensityHistogram);
		kernel_cumulativeHistogram.setArg(1, cumulativeHistogram);
		queue.enqueueNDRangeKernel(kernel_cumulativeHistogram, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange);
		
		// Normalise the cumlative histogram to a maximum value of 255
		cl::Kernel kernel_normaliseHistogram(program, "normalise");
		kernel_normaliseHistogram.setArg(0, cumulativeHistogram);
		kernel_normaliseHistogram.setArg(1, normalisedHistogram);
		queue.enqueueNDRangeKernel(kernel_normaliseHistogram, cl::NullRange, cl::NDRange(256), cl::NullRange);

		// Use the cumulative histogram as a lookup table to map the intensity values to the original image
		cl::Kernel kernel_lookup(program, "lookup");
		kernel_lookup.setArg(0, dev_image_input);
		kernel_lookup.setArg(1, normalisedHistogram);
		kernel_lookup.setArg(2, intensityMap);
		queue.enqueueNDRangeKernel(kernel_lookup, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange);
		
		// 4.3 Copy the result from device to host
		queue.enqueueReadBuffer(intensityMap, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]);

		CImg<unsigned char> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
		CImgDisplay disp_output(output_image,"output");

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
