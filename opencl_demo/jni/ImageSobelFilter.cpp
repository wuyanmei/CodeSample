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

const char* simple_source =
        "__kernel                                           \n"
        "void simpleMultiply(__global float* outputC,       \n"
        "int widthA,                                        \n"
        "int heightA,                                       \n"
        "int widthB,                                        \n"
        "int heightB,                                       \n"
        "__global float* inputA,                            \n"
        "__global float* inputB)                            \n"
        "{                                                  \n"
        "int row=get_global_id(1);                          \n"
        "int col=get_global_id(0);                          \n"
        "float sum=0.0f;                                    \n"
        "for(int i=0;i<widthA;i++){                         \n"
        "sum+=inputA[row*widthA+i]*inputB[i*widthB+col];    \n"
        "}                                                  \n"
        "outputC[row*widthB+col]=sum;                       \n"
        "}";

const char* muti_source =
        "#define BLOCKSIZE 8													\n"
        "__kernel void multMatrix(__global float *mO,							\n"
        "                         int widthA,                                        \n"
        "                         int heightA,                                       \n"
        "                         int widthB,                                        \n"
        "                         int heightB,                                       \n"
        "                         __global float *mA,							\n"
        "                         __global float *mB)				     		\n"
        "{																		\n"
        "	uint lx = get_local_id(0);											\n"
        "	uint ly = get_local_id(1);											\n"
        "	int gx = get_group_id(0);											\n"
        "	int gy = get_group_id(1);											\n"
        "	uint iSubA = BLOCKSIZE * gy * widthA;								\n"
        "	uint iSubB = BLOCKSIZE * gx;										\n"
        "	int n = get_num_groups(0);											\n"
        "	float sum = 0;														\n"
        "	for(int i=0; i< n;i++)												\n"
        "	{																	\n"
        "	  __local float tA[BLOCKSIZE][BLOCKSIZE];							\n"
        "	  __local float tB[BLOCKSIZE][BLOCKSIZE];							\n"
        "	  tA[ly][lx] = mA[ly*widthA + lx + (iSubA + i* BLOCKSIZE)];			\n"
        "	  tB[ly][lx] = mB[ly*widthB + lx + (iSubB + i* BLOCKSIZE * widthB)];\n"
        "	  barrier(CLK_LOCAL_MEM_FENCE);										\n"
        "	  for(int k=0; k<BLOCKSIZE; k++){									\n"
        "	    sum += tA[ly][k] * tB[k][lx];									\n"
        "	  }																	\n"
        "	}																	\n"
        "	int globalIdx=get_global_id(0);										\n"
        "	int globalIdy=get_global_id(1);										\n"
        "	mO[globalIdy * widthA + globalIdx] = sum;							\n"
        "}";

float * simpleMultiply(int len) {
    float*A = NULL;
    float*B = NULL;
    float*C = NULL;
    float*C2 = NULL;
    clock_t matrix_start, matrix_finish;
    cl_int ciErrNum;

    int wA = len, hA = len;
    int wB = len, hB = len;
    int wC = len, hC = len;

    const int elementsA = wA * hA;
    const int elementsB = wB * hB;
    const int elementsC = hA * wB;

    // 计算内存大小
    size_t datasizeA = sizeof(float) * elementsA;
    size_t datasizeB = sizeof(float) * elementsB;
    size_t datasizeC = sizeof(float) * elementsC;
    // 分配内存空间
    A = (float*) malloc(datasizeA);
    B = (float*) malloc(datasizeB);
    C = (float*) malloc(datasizeC);
    C2 = (float*) malloc(datasizeC);

    //init the data
    for (int i = 0; i < wA * hA; i++)
        A[i] = 3.0;

    for (int i = 0; i < wB * hB; i++)
        B[i] = 2.0;

    matrix_start = clock();
    cl_platform_id platform;
    ciErrNum = clGetPlatformIDs(1, &platform, NULL);

    cl_device_id device;
    ciErrNum = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
    cl_context_properties cps[3] = {
            CL_CONTEXT_PLATFORM, (cl_context_properties) platform, 0 };

    cl_context ctx = clCreateContext(cps, 1, &device, NULL, NULL, &ciErrNum);
    cl_command_queue myqueue = clCreateCommandQueue(ctx, device, 0, &ciErrNum);
    cl_mem bufferA = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                                    wA * hA * sizeof(float), NULL, &ciErrNum);
    ciErrNum = clEnqueueWriteBuffer(myqueue, bufferA, CL_TRUE, 0,
                                    wA * hA * sizeof(float), (void*) A, 0, NULL, NULL);
    cl_mem bufferB = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                                    wB * hB * sizeof(float), NULL, &ciErrNum);
    ciErrNum = clEnqueueWriteBuffer(myqueue, bufferB, CL_TRUE, 0,
                                    wB * hB * sizeof(float), (void*) B, 0, NULL, NULL);
    cl_mem bufferC = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                                    hA * wB * sizeof(float), NULL, &ciErrNum);

    cl_program myprog = clCreateProgramWithSource(ctx, 1,
                                                  (const char**) &simple_source, NULL, &ciErrNum);
    ciErrNum = clBuildProgram(myprog, 0, NULL, NULL, NULL, NULL);
    cl_kernel mykernel = clCreateKernel(myprog, "simpleMultiply", &ciErrNum);


    clSetKernelArg(mykernel, 0, sizeof(cl_mem), (void*) &bufferC);
    clSetKernelArg(mykernel, 1, sizeof(cl_int), (void*) &wA);
    clSetKernelArg(mykernel, 2, sizeof(cl_int), (void*) &hA);
    clSetKernelArg(mykernel, 3, sizeof(cl_int), (void*) &wB);
    clSetKernelArg(mykernel, 4, sizeof(cl_int), (void*) &hB);
    clSetKernelArg(mykernel, 5, sizeof(cl_mem), (void*) &bufferA);
    clSetKernelArg(mykernel, 6, sizeof(cl_mem), (void*) &bufferB);

    size_t localws[2] = { 8, 8 };
    size_t globalws[2] = { (size_t)wC, (size_t)hC };

    ciErrNum = clEnqueueNDRangeKernel(myqueue, mykernel, 2, NULL, globalws,
                                      localws, 0, NULL, NULL);
    ciErrNum = clEnqueueReadBuffer(myqueue, bufferC, CL_TRUE, 0,
                                   wC * hC * sizeof(float), (void*) C, 0, NULL, NULL);
    matrix_finish = clock();

    double time_t = (double)(matrix_finish - matrix_start) / CLOCKS_PER_SEC;
    LOGD("Gpu matrix simple multiply - used time: %f s", time_t);
    clReleaseKernel(mykernel);
    clReleaseProgram(myprog);
    clReleaseCommandQueue(myqueue);
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseContext(ctx);

    matrix_start = clock();
    for (int i=0; i < hA; i++) {
        for (int j=0; j < wB; j++) {
            float acc = 0.0f;
            for (int k=0; k < wA; k++) {
                acc += A[i*hA + k] * B[k * wB + k];
            }
            C2[i*hA + j] = acc;
        }
    }
    matrix_finish = clock();
    time_t = (double)(matrix_finish - matrix_start) / CLOCKS_PER_SEC;
    LOGD("Cpu matrix multiply - used time: %f s ", time_t);

    int idx;
    for (idx = 0; idx < wC * hC; idx++) {
        if (C[idx] != C2[idx]) {
            LOGD("matrix multiply err at i=%d C=%f C2=%f", idx, C[idx], C2[idx]);
            break;
        }
    }

    if (idx >= wC * hC) {
        LOGD("matrix multiply - right");
    }


    free(A);
    free(B);
    free(C);
    return C2;
}


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
        "__kernel void vecAdd(  __global float *a,\n"
        "                       __global float *b,\n"
        "                       __global float *c,\n"
        "                       const unsigned int n)\n"
        "{\n"
        "    int id = get_global_id(0);\n"
        "    c[id] = a[id] + b[id];\n"
        "}\n";

float * VectorAddBenchMark(void) {

    clock_t start_t, end_t;


    // Length of vectors
    unsigned int n = 147456;

    // Host input vectors
    float *h_a;
    float *h_b;
    // Host output vector
    float *h_c;
    float *h_d;

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
    size_t bytes = n * sizeof(float);

    // Allocate memory for each vector on host
    h_a = (float *) malloc(bytes);
    h_b = (float *) malloc(bytes);
    h_c = (float *) malloc(bytes);
    h_d = (float *) malloc(bytes);

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
    localSize = 1024;

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
    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);
    double k2 = clock();

    // Read the results from the device
    clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0,
                        bytes, h_c, 0, NULL, NULL);


    // Ends timing since rest is validation and cleanup
    end_t = clock();
    double time_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;
    LOGD("Gpu Vector Add - used time: %f s , kernel use time:%f s", time_t, (k2 - k1)/CLOCKS_PER_SEC);

    for (i = 0; i < n; i++) {
        if(h_c[i] != (h_a[i] + h_b[i])) {
            LOGD("Vector add err at i=%d c=%f a=%f b=%f", i, h_c[i], h_a[i], h_b[i]);
            break;
        }
    }

    if (i >= n) {
        LOGD("Vector Add - success");
    }


    start_t = clock(); // Start Time!
    for (i = 0; i < n; i++) {
        h_d[i] = h_a[i] + h_b[i];
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
    return  h_d;
}

extern "C" {

JNIEXPORT jstring JNICALL Java_com_white_imagesobelfilter_nativeSobelFilter_sobelFilter(
		JNIEnv* env, jobject thiz, jstring imagePath) {

	initOpenCL();

//    VectorAddBenchMark();
    simpleMultiply(1024);
	char rr[100] = "Compute";


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
