#include "com_white_imagesobelfilter_nativeSobelFilter.h"
#include <CL/cl.h>
#include "aopencl.h"
#include <math.h>
#include <stdio.h>
#include <time.h>

#include <android/log.h>
#include <malloc.h>
#include <string.h>

#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, "OpenCLDemo", __VA_ARGS__)

using namespace std;

#define N 20
#define  KERNEL_SRC "\n" \
	"			__kernel void Sobel(__global char *array1, __global char *array2, __global int *array3)		\n "\
	"			{																							\n "\
	"				size_t gidx = get_global_id(0);															\n "\
	"				size_t gidy = get_global_id(1);															\n "\
	"				unsigned char a00, a01, a02;															\n "\
	"				unsigned char a10, a11, a12;															\n "\
	"				unsigned char a20, a21, a22;															\n "\
	"				int width=array3[0];																	\n "\
	"				int heigh=array3[1];																	\n "\
	"				int widthStep=array3[2];																\n "\
	"				if(gidy>0&&gidy<heigh-1&&gidx>0&&gidx<width-1)											\n "\
	"				{																						\n "\
	"					a00 = array1[gidx-1+widthStep*(gidy-1)];											\n "\
	"					a01 = array1[gidx+widthStep*(gidy-1)];												\n "\
	"					a02 = array1[gidx+1+widthStep*(gidy-1)];											\n "\
	"					a10 = array1[gidx-1+widthStep*gidy];												\n "\
	"					a11 = array1[gidx+widthStep*gidy];													\n "\
	"					a12 = array1[gidx+1+widthStep*gidy];												\n "\
	"					a20 = array1[gidx-1+widthStep*(gidy+1)];											\n "\
	"					a21 = array1[gidx+widthStep*(gidy+1)];												\n "\
	"					a22 = array1[gidx+1+widthStep*(gidy+1)];											\n "\
	"					float ux=a20+2*a21+a22-a00-2*a01-a02;												\n "\
	"					float uy=a02+2*a12+a22-a00-2*a10-a20;												\n "\
	"					//array2[gidx+width*gidy]=sqrt(ux*ux + uy*uy);										\n "\
	"					float u=sqrt(ux*ux + uy*uy);														\n "\
	"					if(u>255) {																			\n "\
	"						u=-1;																			\n "\
	"					} else if(u<20) {																	\n "\
	"						u=0;																			\n "\
	"					}																					\n "\
	"					array2[gidx+widthStep*gidy]=u;														\n "\
	"				}																						\n "\
"}"


void dumpCLInfo()
{
    cl_int status;
    cl_uint numPlatforms;
    cl_platform_id* platforms;
    cl_uint numDevices = 0;
    cl_device_id *devices;

	LOGD("*** OpenCL info ***");
	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	/*For clarity, choose the first available platform. */
	if (numPlatforms <= 0 || status != CL_SUCCESS) {
        LOGD("opencl_process fail get platform numPlatforms=%d\n", numPlatforms);
        return;
    }

	platforms = (cl_platform_id*) malloc(numPlatforms * sizeof(cl_platform_id));
	status = clGetPlatformIDs(numPlatforms, platforms, NULL);
    if (numPlatforms <= 0 || status != CL_SUCCESS) {
        LOGD("fail get platform IDS\n");
        return;
    }

    LOGD("OpenCL info: Found %d OpenCL platforms", numPlatforms);
    for (int i = 0; i < numPlatforms; i++) {
        char name[256];
        char vendor[256];
        char version[256];
        char profile[256];
        char extension[1024];
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 256, &name[0], NULL);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 256, &vendor[0], NULL);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, 256, &version[0], NULL);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_PROFILE, 256, &profile[0], NULL);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_EXTENSIONS, 1024, &extension[0], NULL);
        LOGD( "OpenCL info: Platform[%d] = %s\n\t %s\n\t %s\n\t %s\n\t %s",
              i, name, version, profile, vendor, extension);

        /*Step 2:Query the platform and choose the first GPU device if has one.Otherwise use the CPU as device.*/
        status = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
        if (numDevices <= 0 || status != CL_SUCCESS) {
            LOGD("No GPU device available at platform[%d]\n", i);
            continue;
        }

        devices = (cl_device_id*) malloc(numDevices * sizeof(cl_device_id));
        status = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
        if (status != CL_SUCCESS) {
            LOGD("fail get GetDeviceIDs\n");
            return;
        }

        for (int j = 0; j < numDevices; ++j) {
            char name[256];
            cl_ulong type;
            char extensions[1024];
            clGetDeviceInfo(devices[0], CL_DEVICE_NAME, 256, name, NULL);
            clGetDeviceInfo(devices[0], CL_DEVICE_TYPE, sizeof(type), &type, NULL);
            clGetDeviceInfo(devices[0], CL_DEVICE_EXTENSIONS, 1024, extensions, NULL);
			LOGD( "OpenCL info: Device[%d] = %s (%s) \n\t ext = %s",
				  i, name, (type==CL_DEVICE_TYPE_GPU ? "GPU" : "CPU"), extensions);
        }

    }

	LOGD("*******************");
}


void initOpenCL() {
	LOGD("initOpenCL 00000\n");
	initFns();
	dumpCLInfo();
}

const char *kernel_vadd =
        "__kernel void vecAdd(  __global int *a,\n"
        "                       __global int *b,\n"
        "                       __global int *c,\n"
        "                       const unsigned int n)\n"
        "{\n"
        "    int id = get_global_id(0);\n"
        "    c[id] = a[id] + b[id];\n"
        "}\n";

void VectorAddByGpu(void) {

    clock_t start_t, end_t;


    // Length of vectors
    unsigned int n = 147456;

    // Host input vectors
    int *h_a;
    int *h_b;
    // Host output vector
    int *h_c;

    // Device input buffers
    cl_mem d_a;
    cl_mem d_b;
    // Device output buffer
    cl_mem d_c;

    cl_platform_id cpPlatform;        // OpenCL platform
    cl_device_id device_id;           // device ID
    cl_context context;               // context
    cl_command_queue queue;           // command queue
    cl_program program;               // program
    cl_kernel kernel;                 // kernel

    // Size, in bytes, of each vector
    size_t bytes = n * sizeof(int);

    // Allocate memory for each vector on host
    h_a = (int *) malloc(bytes);
    h_b = (int *) malloc(bytes);
    h_c = (int *) malloc(bytes);

    // Initialize vectors on host
    int i;
    for (i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i + i;
        h_c[i] = 0;
    }

    size_t globalSize, localSize;
    cl_int err;
    // Number of work items in each local work group
    localSize = 64;

    // Number of total work items - localSize must be devisor
    globalSize = n;

    // Bind to platform
    err = clGetPlatformIDs(1, &cpPlatform, NULL);

    // Get ID for the device
    err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

    // Create a context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

    // Create a command queue
    queue = clCreateCommandQueue(context, device_id, 0, &err);

    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1,
                                        (const char **) &kernel_vadd, NULL, &err);

    // Build the program executable
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, "vecAdd", &err);

    start_t = clock(); // Start Time!
    // Create the input and output arrays in device memory for our calculation
    d_a = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, bytes, h_a, NULL);
    d_b = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, bytes, h_b, NULL);
    d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);


    // Set the arguments to our compute kernel
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &n);

    double k1 = clock();
    // Execute the kernel over the entire range of the data set
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize,
                                 0, NULL, NULL);
    double k2 = clock();

    // Wait for the command queue to get serviced before reading back results
//    clFinish(queue);

    // Read the results from the device
    clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0,
                        bytes, h_c, 0, NULL, NULL);


    // Ends timing since rest is validation and cleanup
    end_t = clock();
    double time_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;
    LOGD("Gpu Vector Add - used time: %f s , kernel use time:%f s", time_t, (k2 - k1)/CLOCKS_PER_SEC);

    for (i = 0; i < n; i++) {
        if(h_c[i] != (h_a[i] + h_b[i])) {
            LOGD("Vector add err at i=%d c=%d a=%d b=%d", i, h_c[i], h_a[i], h_b[i]);
            break;
        }
    }

    if (i >= n) {
        LOGD("Vector Add - success");
    }


    start_t = clock(); // Start Time!
    for (i = 0; i < n; i++) {
        h_c[i] = h_a[i] + h_b[i];
    }
    end_t = clock();
    time_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;
    LOGD("Cpu Vector Add - used time: %f s", time_t);

    // release OpenCL resources
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    //release host memory
    free(h_a);
    free(h_b);
    free(h_c);
}

extern "C" {

JNIEXPORT jstring JNICALL Java_com_white_imagesobelfilter_nativeSobelFilter_sobelFilter(
		JNIEnv* env, jobject thiz, jstring imagePath) {

	initOpenCL();

    VectorAddByGpu();

	char rr[100] = "Compute result:\nGPU:";


	const char* result = rr;
	return env->NewStringUTF(rr);
}
}



//void opencl_process(void) {
//    LOGD("opencl_process 00000\n");
//	initFns();
//	/* */
//	cl_uint numPlatforms; //the NO. of platforms
//	cl_platform_id platform = NULL; //the chosen platform
//	cl_int status;
//	cl_platform_id* platforms;
//	cl_uint numDevices = 0;
//	cl_device_id *devices;
//	cl_context context;
//	cl_command_queue commandQueue;
//	cl_program program;
//	cl_kernel kernel;
//	//size_t global;
//	cl_mem a1, a2, a3;
//    LOGD("opencl_process 11111\n");
//
//	char *inputData1;
//
//
//	/*Step1: Getting platforms and choose an available one.*/
//    LOGD("opencl_process 22222 clGetPlatformIDs=%p\n", clGetPlatformIDs);
//	status = clGetPlatformIDs(0, NULL, &numPlatforms);
//    LOGD("opencl_process 33333 numPlatforms=%d\n", numPlatforms);
//	/*For clarity, choose the first available platform. */
//	if (numPlatforms > 0) {
//		platforms = (cl_platform_id*) malloc(
//				numPlatforms * sizeof(cl_platform_id));
//		status = clGetPlatformIDs(numPlatforms, platforms, NULL);
//		platform = platforms[0];
//		free(platforms);
//	}
//
//	/*Step 2:Query the platform and choose the first GPU device if has one.Otherwise use the CPU as device.*/
//	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
//	if (numDevices == 0) {
//		//no GPU available.
//        LOGD("No GPU device available.\n");
//        LOGD("Choose CPU as default device.\n");
//		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL,
//				&numDevices);
//		devices = (cl_device_id*) malloc(numDevices * sizeof(cl_device_id));
//		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, numDevices,
//				devices, NULL);
//	} else {
//		devices = (cl_device_id*) malloc(numDevices * sizeof(cl_device_id));
//		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices,
//				devices, NULL);
//	}
//	devices = (cl_device_id*) malloc(numDevices * sizeof(cl_device_id));
//	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices,
//			NULL);
//
//	/*Step 3: Create context.*/
//	context = clCreateContext(NULL, 1, devices, NULL, NULL, &status);
//
//	/*Step 4: Creating command queue associate with the context.*/
//	commandQueue = clCreateCommandQueue(context, devices[0], 0, &status);
//
//	/*Step 5: Create program object */
//	const char *source = KERNEL_SRC;
//	size_t sourceSize[] = { strlen(source) };
//	program = clCreateProgramWithSource(context, 1, &source, sourceSize,
//			&status);
//
//	/*Step 6: Build program. */
//	status = clBuildProgram(program, 1, devices, NULL, NULL, NULL);
//
//	a1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
//			sizeof(unsigned char) * IN_DATA_SIZE, inputData1, &status);
//
//	a2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
//			sizeof(unsigned char) * OUT_DATA_SIZE, outputData, &status);
//
//	a3 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
//			sizeof(int) * 3, inputData2, &status);
//
//	/*Step 8: Create kernel object */
//	kernel = clCreateKernel(program, "Sobel", &status);
//
//	// set the argument list for the kernel command
//	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a1);
//	status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &a2);
//	status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &a3);
//
//	size_t local[] = { 11, 11 };
//	size_t global[2];
//	global[0] = (
//			IMG_WIDTH % local[0] == 0 ?
//					IMG_WIDTH : (IMG_WIDTH + local[0] - IMG_WIDTH % local[0]));
//	global[1] =
//			(IMG_HEIGHT % local[1] == 0 ?
//					IMG_HEIGHT : (IMG_HEIGHT + local[1] - IMG_HEIGHT % local[1]));
//
//	status = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, global,
//			local, 0, NULL, NULL);
//	if (status != 0)
//		return;
//	clFinish(commandQueue);
//
//	clEnqueueReadBuffer(commandQueue, a2, CL_TRUE, 0,
//			sizeof(unsigned char) * OUT_DATA_SIZE, outputData, 0, NULL, NULL);
//
//	clReleaseMemObject(a1);
//	clReleaseMemObject(a2);
//	clReleaseMemObject(a3);
//	clReleaseProgram(program);
//	clReleaseKernel(kernel);
//	clReleaseCommandQueue(commandQueue);
//	clReleaseContext(context);
//
//}
