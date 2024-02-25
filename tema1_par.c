// Author: APD team, except where source was noted

#include "helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>

#define CONTOUR_CONFIG_COUNT    16
#define FILENAME_MAX_SIZE       50
#define STEP                    8
#define SIGMA                   200
#define RESCALE_X               2048
#define RESCALE_Y               2048

#define CLAMP(v, min, max) if(v < min) { v = min; } else if(v > max) { v = max; }
#define min(a, b)(a > b ? b : a)


/* Thread structure that stores its basic information(id, numThreads), the barrier shared by all threads and
 the image(initial and rescaled) and grid. */
struct args_thread {
	int id;
	int numThreads;
	int step_x;
	int step_y;
	unsigned char **grid;
	ppm_image *image;
	ppm_image *scaled_image;
	ppm_image **contour_map;
	pthread_barrier_t *b;
};

// Creates a map between the binary configuration (e.g. 0110_2) and the corresponding pixels
// that need to be set on the output image. An array is used for this map since the keys are
// binary numbers in 0-15. Contour images are located in the './contours' directory.
ppm_image **init_contour_map() {
	ppm_image **map = (ppm_image **)malloc(CONTOUR_CONFIG_COUNT * sizeof(ppm_image *));
	if (!map) {
		fprintf(stderr, "Unable to allocate memory\n");
		exit(1);
	}

	for (int i = 0; i < CONTOUR_CONFIG_COUNT; i++) {
		char filename[FILENAME_MAX_SIZE];
		sprintf(filename, "./contours/%d.ppm", i);
		map[i] = read_ppm(filename);
	}

	return map;
}

// Updates a particular section of an image with the corresponding contour pixels.
// Used to create the complete contour image.
void update_image(ppm_image *scaled_image, ppm_image *contour, int x, int y) {
	for (int i = 0; i < contour->x; i++) {
		for (int j = 0; j < contour->y; j++) {
			int contour_pixel_index = contour->x * i + j;
			int image_pixel_index = (x + i) * scaled_image->y + y + j;

			scaled_image->data[image_pixel_index].red = contour->data[contour_pixel_index].red;
			scaled_image->data[image_pixel_index].green = contour->data[contour_pixel_index].green;
			scaled_image->data[image_pixel_index].blue = contour->data[contour_pixel_index].blue;
		}
	}
}

/* Allocate space for the new image(rescaled one) if its dimensions go above the limit, image shared
 with all the threads. If not, it returns the initial image.*/
ppm_image *rescale_init(ppm_image* image) {
	// we only rescale downwards
	if (image->x <= RESCALE_X && image->y <= RESCALE_Y) {
		return image;
	}

	// alloc memory for image
	ppm_image *new_image = (ppm_image *)malloc(sizeof(ppm_image));
	if (!new_image) {
		fprintf(stderr, "Unable to allocate memory\n");
		exit(1);
	}
	// Set the maximum sizes
	new_image->x = RESCALE_X;
	new_image->y = RESCALE_Y;

	new_image->data = (ppm_pixel*)malloc(new_image->x * new_image->y * sizeof(ppm_pixel));
	if (!new_image) {
		fprintf(stderr, "Unable to allocate memory\n");
		exit(1);
	}

	return new_image;
}

/* Function that rescales an image using the sample_bicubic. By using threads, the array is divided
 into sections and worked simultaneously. */
void rescale_image(struct args_thread* argst) {
	// If the sizes of the original image respect the maximum edges(sizes), then the original
	// image is kept, this made already made in rescale_init
	if (argst->image->x <= RESCALE_X && argst->image->y <= RESCALE_Y) {
		return;
	}

	uint8_t sample[3];

	/* Divide the array by sections for each thread to work seperately.*/
	int start = argst->id * (double)argst->scaled_image->x / argst->numThreads;
	int end = min((argst->id + 1) * (double)argst->scaled_image->x / argst->numThreads, argst->scaled_image->x);

	// use bicubic interpolation for scaling
	for (int i = start; i < end; i++) {
		for (int j = 0; j < argst->scaled_image->y; j++) {
			float u = (float)i / (float)(argst->scaled_image->x - 1);
			float v = (float)j / (float)(argst->scaled_image->y - 1);
			sample_bicubic(argst->image, u, v, sample);

			argst->scaled_image->data[i * argst->scaled_image->y + j].red = sample[0];
			argst->scaled_image->data[i * argst->scaled_image->y + j].green = sample[1];
			argst->scaled_image->data[i * argst->scaled_image->y + j].blue = sample[2];
		}
	}
	// Wait for all the threads to finish the task.
	pthread_barrier_wait(argst->b);
}

/* Function that allocates space for the grid of the image. */
unsigned char **sample_grid_init(ppm_image *scaled_image, int step_x, int step_y){
	int p = scaled_image->x / step_x;
	int q = scaled_image->y / step_y;

	unsigned char **grid = (unsigned char **)malloc((p + 1) * sizeof(unsigned char*));
	if (!grid) {
		fprintf(stderr, "Unable to allocate memory\n");
		exit(1);
	}

	for (int i = 0; i <= p; i++) {
		grid[i] = (unsigned char *)malloc((q + 1) * sizeof(unsigned char));
		if (!grid[i]) {
			fprintf(stderr, "Unable to allocate memory\n");
			exit(1);
		}
	}

	return grid;
}

// Corresponds to step 1 of the marching squares algorithm, which focuses on sampling the image.
// Builds a p x q grid of points with values which can be either 0 or 1, depending on how the
// pixel values compare to the `sigma` reference value. The points are taken at equal distances
// in the original image, based on the `step_x` and `step_y` arguments.
void sample_grid(struct args_thread* argst, unsigned char sigma) {
	int p = argst->scaled_image->x / argst->step_x;
	int q = argst->scaled_image->y / argst->step_y;
	
	/* Divide the matrix by the numbers of lines for each thread to work seperately.*/
	int start = argst->id * (double)p / argst->numThreads;
	int end = min((argst->id + 1) * (double)p / argst->numThreads, p);

	for (int i = start; i < end; i++) {
		for (int j = 0; j < q; j++) {
			ppm_pixel curr_pixel = argst->scaled_image->data[i * argst->step_x * argst->scaled_image->y + j * argst->step_y];

			unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

			if (curr_color > sigma) {
				argst->grid[i][j] = 0;
			} else {
				argst->grid[i][j] = 1;
			}
		}
	}
	// Wait for all the threads to finish in order to have all the values
	pthread_barrier_wait(argst->b);

	// last sample points have no neighbors below / to the right, so we use pixels on the
	// last row / column of the input image for them
	for (int i = start; i < end; i++) {
		ppm_pixel curr_pixel = argst->scaled_image->data[i * argst->step_x * argst->scaled_image->y + argst->scaled_image->x - 1];

		unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

		if (curr_color > sigma) {
			argst->grid[i][q] = 0;
		} else {
			argst->grid[i][q] = 1;
		}
	}
	pthread_barrier_wait(argst->b);

	// Now, it is divided by columns in order to parallelize. 
	start = argst->id * (double)q / argst->numThreads;
	end = min((argst->id + 1) * (double)q / argst->numThreads, q);

	for (int j = start; j < end; j++) {
		ppm_pixel curr_pixel = argst->scaled_image->data[(argst->scaled_image->x - 1) * argst->scaled_image->y + j * argst->step_y];

		unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

		if (curr_color > sigma) {
			argst->grid[p][j] = 0;
		} else {
			argst->grid[p][j] = 1;
		}
	}

	pthread_barrier_wait(argst->b);
}

// Corresponds to step 2 of the marching squares algorithm, which focuses on identifying the
// type of contour which corresponds to each subgrid. It determines the binary value of each
// sample fragment of the original image and replaces the pixels in the original image with
// the pixels of the corresponding contour image accordingly.
void march(struct args_thread* argst) {
	int p = argst->scaled_image->x / argst->step_x;
	int q = argst->scaled_image->y / argst->step_y;

	// Again, each thread works on different lines in parallel
	int start = argst->id * (double)p / argst->numThreads;
	int end = min((argst->id + 1) * (double)p / argst->numThreads, p);

	for (int i = start; i < end; i++) {
		for (int j = 0; j < q; j++) {
			unsigned char k = 8 * argst->grid[i][j] + 4 * argst->grid[i][j + 1] + 2 * argst->grid[i + 1][j + 1] + 1 * argst->grid[i + 1][j];
			update_image(argst->scaled_image, argst->contour_map[k], i * argst->step_x, j * argst->step_y);
		}
	}
	
	pthread_barrier_wait(argst->b);
}

// Calls `free` method on the utilized resources.
void free_resources(ppm_image *image, ppm_image **contour_map, unsigned char **grid, int step_x) {
	for (int i = 0; i < CONTOUR_CONFIG_COUNT; i++) {
		free(contour_map[i]->data);
		free(contour_map[i]);
	}
	free(contour_map);

	for (int i = 0; i <= image->x / step_x; i++) {
		free(grid[i]);
	}
	free(grid);
}

void *resolver(void *arg) {
	struct args_thread* argst = (struct args_thread*) arg;

	// 1. Rescale the image
	rescale_image(argst);

	// 2. Sample the grid
	sample_grid(argst, SIGMA);

	// 3. March the squares
	march(argst);
}



int main(int argc, char *argv[]) {
	if (argc < 4) {
		fprintf(stderr, "Usage: ./tema1 <in_file> <out_file> <P>\n");
		return 1;
	}

	int numThreads = atoi(argv[3]);
	ppm_image *image = read_ppm(argv[1]);

	// Create an array of thread structure in order to store for each thread its
	// informations, ans allocate space
	struct args_thread* argst;
	argst = (struct args_thread *)malloc(numThreads * sizeof(struct args_thread));

	// Allocate space for the array of threads with individual IDs
	pthread_t* threads;
	threads = (pthread_t*) malloc(numThreads * sizeof(pthread_t));

	// Create the barrier that stops the threads from continuing executing until all
	// the threads finished the assigned section of code
	pthread_barrier_t barrier;
	pthread_barrier_init(&barrier, NULL, numThreads);

	// 0. Initialize contour map
	ppm_image **contour_map = init_contour_map();

	// Initialize scaled image
	ppm_image *scaled_image = rescale_init(image);

	// Initialize the grid
	unsigned char **grid = sample_grid_init(scaled_image, STEP, STEP);

	int freeImage = 0;

	for (int i = 0; i < numThreads; i++) {
		// Save each characteristic for each thread
		argst[i].id = i;
		argst[i].numThreads = numThreads;
		argst[i].step_x = STEP;
		argst[i].step_y = STEP;
		argst[i].image = image;
		// Common work of space
		argst[i].b = &barrier;
		argst[i].contour_map = contour_map;
		argst[i].scaled_image = scaled_image;
		argst[i].grid = grid;
		
		// Calls the function that each thread executes
		int r = pthread_create(&threads[i], NULL, resolver, &argst[i]);

		if (r) {
			printf("Eroare la crearea thread-ului %d\n", i);
			exit(-1);
		}

		// If the image is rescaled, the initial image needs to be freed.
		if (argst[i].scaled_image->x == RESCALE_X) {
			freeImage = 1;
		}
	}

	for (int i = 0; i < numThreads; i++) {
		// Join all the threads so as the main thread is the single one remained
		int r = pthread_join(threads[i], NULL);

		if (r) {
			printf("Eroare la asteptarea thread-ului %d\n", i);
			exit(-1);
		}
	}

	// Destroy the barrier
	pthread_barrier_destroy(&barrier);

	// 4. Write output
	write_ppm(scaled_image, argv[2]);

	// Free all resources
	free_resources(scaled_image, contour_map, grid, STEP);
	free(argst);
	if (freeImage == 1) {
		free(image->data);
		free(image);
	}
	
	return 0;
}
