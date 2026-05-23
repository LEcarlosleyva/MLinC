#pragma once


#include <stdlib.h>
#include <stdint.h>
#define ARENA_INIT_SIZE 4000000

typedef struct {

	void* arena_space_ptr;
	void* arena_space_current;
	size_t arena_space_occupied;
	size_t arena_space_left;
	size_t id;

} Arena;


void arena_init(Arena* arena, int size);
void* arena_alloc(Arena* arena, size_t size);
void arena_free(Arena* arena);

