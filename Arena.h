#pragma once


#include <stdlib.h>
#include <stdint.h>
//#define ARENA_INIT_SIZE 4096
#define ARENA_INIT_SIZE 4096*3
#define ARENA_PADDING_SIZE 16
typedef struct {

	void* arena_space_ptr;
	void* arena_space_current;
	size_t arena_space_occupied;
	size_t arena_space_left;
	size_t id;

} Arena;


Arena * arena_init( int size);
void* arena_alloc(Arena* arena, size_t size);
void arena_free(Arena** arena);
void arena_init_st(Arena* arena, int size);
void arena_report(Arena arena);

