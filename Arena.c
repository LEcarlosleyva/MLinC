

#include "Arena.h"




static int id_counter = 0;


void arena_init_st(Arena* arena, int size) {

	if (!arena) return;

	int arena_size = 0;

	if (size == 0) {
		arena_size = ARENA_INIT_SIZE;
	}
	else {
		arena_size = size;
	}

	arena->arena_space_ptr = malloc(arena_size);
	if (!arena->arena_space_ptr) return;
	arena->arena_space_current = arena->arena_space_ptr;
	arena->arena_space_occupied = 0;
	arena->arena_space_left = arena_size;
	arena->id = id_counter++;


}

Arena * arena_init(int size) {
	
	Arena* arena = malloc(sizeof(Arena));

	if (!arena) {
		printf("failed to allocate space for arena in arena_init\n");
		return NULL;
	}
		
		
	
	int arena_size = 0;

	if (size == 0) {
		arena_size = ARENA_INIT_SIZE;
	}
	else {
		arena_size = size;
	}

	arena->arena_space_ptr = malloc(arena_size);
	if (!arena->arena_space_ptr) {
		printf("failed to allocate space for arena in arena_init\n");
		return NULL;
	}
	arena->arena_space_current = arena->arena_space_ptr;
	arena->arena_space_occupied = 0;
	arena->arena_space_left = arena_size;
	arena->id = id_counter++;



	return arena;

}


void* arena_alloc(Arena* arena, size_t size) {

	if (!arena) {
		printf("pointer to arrena is null in arena_alloc\n");
		return NULL;
	}
	

	
	size_t padding = (ARENA_PADDING_SIZE - size % ARENA_PADDING_SIZE)%ARENA_PADDING_SIZE;



	if (size + padding > arena->arena_space_left) {
		printf("no more space in the current arena in arena alloc : TODO implment realloc\n");
		return NULL;
	}

	void* ptr = arena->arena_space_current;

	

	uint8_t * tmp = (uint8_t *) arena->arena_space_current + size + padding;

	arena->arena_space_current = (void*)tmp;

	arena->arena_space_occupied += size + padding;
	arena->arena_space_left -= size + padding;

	return ptr;


}

void arena_report(Arena arena) {


	printf("arena #: %d space occupied is : %d and the space left is : %d \n",arena.id,arena.arena_space_occupied,arena.arena_space_left );

}

void arena_free(Arena** arena) {

	free((*arena)->arena_space_ptr);
	free(*arena);
	
}