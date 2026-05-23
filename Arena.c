

#include "Arena.h"




static int id_counter = 0;




void arena_init(Arena * arena,int size) {
	if (!arena) return;
	if (size == 0) {
		arena->arena_space_ptr = malloc(ARENA_INIT_SIZE);
		if (!arena->arena_space_ptr) return;
		arena->arena_space_current = arena->arena_space_ptr;
		arena->arena_space_occupied = 0;
		arena->arena_space_left = ARENA_INIT_SIZE;
		arena->id = id_counter++;



	}
	else {
		arena->arena_space_ptr = malloc(size);
		if (!arena->arena_space_ptr) return;
		arena->arena_space_current = arena->arena_space_ptr;
		arena->arena_space_occupied = 0;
		arena->arena_space_left = size;
		arena->id = id_counter++;

	}




}


void* arena_alloc(Arena * arena,size_t size) {
	if (size > arena->arena_space_left) return NULL;

	void* ptr = arena->arena_space_current;

	(uint8_t *) arena->arena_space_current += size;
	arena->arena_space_occupied += size;
	arena->arena_space_left -= size;

	return ptr;


}



void arena_free(Arena* arena) {

	free(arena->arena_space_ptr);
}